#!/usr/bin/env python
"""
Milestone 1 - Early Parkinson's Detection Using Speech Analysis
LSTM-based spectrogram analysis per Rahmatallah et al. (2025), integrated into our milestone1_skeleton pipeline.
- Uses mel-scale spectrograms matching paper: 256 mel bands, 1.5s central segment, 90% overlap, window=512, n_fft=1024.
- Feeds sequence of mel frames into LSTM classifier.
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

SAMPLE_RATE = 16000
SEGMENT_DURATION = 1.5  # seconds per paper
N_FFT = 1024
WIN_LENGTH = 512         # 32 ms
HOP_LENGTH = int(WIN_LENGTH * 0.1)  # 90% overlap
N_MELS = 256
FMIN = 50
FMAX = 4000

SEGMENT_FRAMES = int(np.ceil(SEGMENT_DURATION * SAMPLE_RATE / HOP_LENGTH))

RANDOM_SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count() or 2
# ------------------------------------------------

# -------- Data utilities --------

def cache_all(plot: bool = False):
    """Precompute and verify spectrogram files (audio or image mode)."""
    # In image mode, assume images are already generated.
    # You can extend this to copy or check files if needed.
    print("[cache_all] (image mode) Skipped pre-caching")


def list_wav_files() -> List[Tuple[Path, int]]:
    wavs = []
    for label_name, label in [("HC_AH", 0), ("PD_AH", 1)]:
        for wav in (DATA_ROOT / label_name).glob("*.wav"):
            wavs.append((wav, label))
    return wavs


def load_and_preprocess(path: Path) -> np.ndarray:
    # Load and normalize
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    y = librosa.util.normalize(y)
    # Compute mel-spectrogram
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    logmel = librosa.power_to_db(melspec, ref=np.max)
    logmel = logmel.T.astype(np.float32)  # shape (T, M)
    T, M = logmel.shape
    # Center 1.5s segment in frames
    if T >= SEGMENT_FRAMES:
        start = (T - SEGMENT_FRAMES) // 2
        seg = logmel[start:start + SEGMENT_FRAMES]
    else:
        # pad if shorter
        pad_top = (SEGMENT_FRAMES - T) // 2
        pad_bot = SEGMENT_FRAMES - T - pad_top
        seg = np.pad(logmel, ((pad_top, pad_bot), (0, 0)), mode='constant', constant_values=-80.0)
    return seg  # (SEGMENT_FRAMES, N_MELS)

# -------- Dataset & Augmentation --------
class ParkinsonDataset(Dataset):
    def __init__(self,
                 files: List[Path],
                 labels: List[int],
                 augment: bool = False,
                 use_image: bool = False):
        self.files = files
        self.labels = labels
        self.augment = augment
        self.use_image = use_image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        if self.use_image:
            mel = load_image_spectrogram(path)
        else:
            mel = load_and_preprocess(path)
        if self.augment and not self.use_image:
            # audio-based augmentations on spectrogram
            T, _ = mel.shape
            # time warping
            rate = 1.0 + np.random.uniform(-0.2, 0.2)
            warped = librosa.effects.time_stretch(mel.T, rate=rate).T
            if warped.shape[0] < T:
                warped = np.pad(warped, ((0, T - warped.shape[0]), (0, 0)), mode='constant', constant_values=warped.mean())
            mel = warped[:T]
            # spec augment
            for _ in range(2):
                t = np.random.randint(0, int(T * 0.1)); t0 = np.random.randint(0, T - t)
                mel[t0:t0+t, :] = mel.mean()
            for _ in range(2):
                f = np.random.randint(0, int(mel.shape[1] * 0.1)); f0 = np.random.randint(0, mel.shape[1] - f)
                mel[:, f0:f0+f] = mel.mean()
        mel = mel.astype(np.float32)
        return torch.from_numpy(mel), torch.tensor(self.labels[idx], dtype=torch.float32)

# -------- Model --------
class LSTMAudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=N_MELS,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        d = 128 * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(0.3),
            nn.Linear(d, d // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(d // 4, 1),
        )

    def forward(self, x):  # x: (B, T, M)
        out, _ = self.lstm(x)
        h = out.mean(dim=1)
        return self.classifier(h).squeeze(1)

# -------- Training Helpers --------
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
    return total_loss/total, correct/total, np.array(ys), np.array(ps)

# -------- Main Entry --------
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

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_dl, criterion, device)
        print(f"Epoch {ep:02d} | train_loss {tr_loss:.3f} | train_acc {tr_acc:.3f} "
              f"| val_loss {val_loss:.3f} | val_acc {val_acc:.3f}")

    _, _, y_true, y_prob = evaluate(model, val_dl, criterion, device)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    best_thr = thr[np.argmax(tpr - fpr)]
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

