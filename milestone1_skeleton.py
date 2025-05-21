#!/usr/bin/env python
"""
Milestone 1 - Early Parkinson's Detection Using Speech Analysis
Data pipeline & experimental skeleton code + on-the-fly mel-spectrogram augmentation
with balanced sampler, weighted loss, epoch training/validation loss & acc, and final metrics
------------------------------------------------
usage: milestone1_skeleton.py [options]
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import librosa
import warnings
warnings.filterwarnings("ignore", message="n_fft=.* is too large for input signal of length.*")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# -------------------- CONFIG --------------------
DATA_ROOT = Path("data-source/audio")
CACHE_DIR = Path("artifacts/mel_specs")
PLOT_DIR = Path("artifacts/plots")
CHECKPOINT_DIR = Path("artifacts/checkpoints")
STATS_DIR = Path("artifacts/stats")

SAMPLE_RATE = 16_000       # 16 kHz
N_MELS = 64
HOP_LENGTH = 160           # 10 ms
WIN_LENGTH = 400           # 25 ms
FMIN = 50
FMAX = 4_000               # adapt to original
N_FFT = 512                # FFT window size
MAX_FRAMES = 1_024         # ≈10 s @ 100 fps

RANDOM_SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count() or 2
# ------------------------------------------------

# -------- Data utilities --------
def list_wav_files() -> List[Tuple[Path, int]]:
    wavs = []
    for label_name, label in [("HC_AH", 0), ("PD_AH", 1)]:
        for wav in (DATA_ROOT / label_name).glob("*.wav"):
            wavs.append((wav, label))
    return wavs


def load_and_preprocess(path: Path) -> np.ndarray:
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    y = librosa.util.normalize(y)
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    logmel = librosa.power_to_db(melspec, ref=np.max).T.astype(np.float32)
    T, _ = logmel.shape
    if T < MAX_FRAMES:
        logmel = np.pad(logmel, ((0, MAX_FRAMES - T), (0, 0)), mode='constant', constant_values=-80.0)
    else:
        logmel = logmel[:MAX_FRAMES]
    return logmel


def cache_all(plot: bool = False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for wav, _ in list_wav_files():
        out = CACHE_DIR / f"{wav.parent.name.split('_')[0]}_{wav.stem}.npy"
        if not out.exists():
            spec = load_and_preprocess(wav)
            np.save(out, spec)
            if plot:
                plt.figure(figsize=(10,4))
                plt.imshow(spec.T, aspect='auto', origin='lower')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f"Log-mel spectrogram of {wav.stem}")
                plt.savefig(PLOT_DIR / f"{wav.stem}.png")
                plt.close()
    print("[cache_all] Done caching mel-spectrograms")

# -------- Dataset & Augmentation --------
class ParkinsonDataset(Dataset):
    def __init__(self, wavs: List[Path], labels: List[int], augment: bool = False):
        self.wavs = wavs
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        mel = load_and_preprocess(self.wavs[idx])
        if self.augment:
            # Time warping
            T, _ = mel.shape
            rate = 1.0 + np.random.uniform(-0.2, 0.2)
            warped = librosa.effects.time_stretch(mel.T, rate=rate).T
            if warped.shape[0] < T:
                warped = np.pad(warped, ((0, T - warped.shape[0]), (0, 0)), mode='constant', constant_values=warped.mean())
            mel = warped[:T]
            # SpecAugment: time & freq masking
            for _ in range(2):
                t = np.random.randint(0, int(T * 0.1)); t0 = np.random.randint(0, T - t)
                mel[t0:t0+t, :] = mel.mean()
            for _ in range(2):
                f = np.random.randint(0, int(N_MELS * 0.1)); f0 = np.random.randint(0, N_MELS - f)
                mel[:, f0:f0+f] = mel.mean()
            # Noise
            dyn = mel.max() - mel.min(); mel += np.random.randn(*mel.shape) * (0.02 * dyn)
            # Time shift
            shift = np.random.randint(-int(T * 0.1), int(T * 0.1)); mel = np.roll(mel, shift, axis=0)
        mel = mel.astype(np.float32)
        return torch.from_numpy(mel), torch.tensor(self.labels[idx], dtype=torch.float32)

# -------- Model --------
class LSTMAudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_MELS, 128, num_layers=1, batch_first=True, bidirectional=True)
        d = 128 * 2
        self.fc = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(0.3),
            nn.Linear(d, d // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(d // 4, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out.mean(dim=1)
        return self.fc(h).squeeze(1)

# -------- Helpers --------
def train_one_epoch(model, dl, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
        correct += (preds == y.cpu().numpy()).sum()
        total += y.size(0)
    return total_loss / total, correct / total


def evaluate(model, dl, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            ys.extend(y.cpu().numpy())
            ps.extend(probs)
            correct += (preds == y.cpu().numpy()).sum()
            total += y.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    y_true = np.array(ys)
    y_prob = np.array(ps)
    return avg_loss, accuracy, y_true, y_prob

# -------- Main --------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    print(f"START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    cache_all(plot=args.plot)

    pairs = list_wav_files()
    wavs, labels = zip(*pairs)
    train_w, val_w, train_l, val_l = train_test_split(
        wavs, labels, test_size=0.3, stratify=labels, random_state=RANDOM_SEED
    )

    counts = np.bincount(train_l)
    class_weights = 1.0 / counts
    sample_weights = class_weights[train_l]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_dl = DataLoader(
        ParkinsonDataset(train_w, list(train_l), augment=True),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
    )
    val_dl = DataLoader(
        ParkinsonDataset(val_w, list(val_l), augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMAudioClassifier().to(device)

    pos_weight = torch.tensor(counts[0] / counts[1], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop with logging
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_dl, criterion, device)
        print(
            f"Epoch {ep:02d} | train_loss {tr_loss:.3f} | train_acc {tr_acc:.3f} "
            f"| val_loss {val_loss:.3f} | val_acc {val_acc:.3f}"
        )

    # Final metrics
    _, _, y_true, y_prob = evaluate(model, val_dl, criterion, device)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    idx = np.argmax(tpr - fpr)
    best_thr = thr[idx]
    y_pred = (y_prob > best_thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print("--- FINAL METRICS ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1 score : {f1:.3f}")
    print(f"ROC AUC  : {auc:.3f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == '__main__':
    main()
