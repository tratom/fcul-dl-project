#!/usr/bin/env python
"""
Milestone 1 - Early Parkinson's Detection Using Speech Analysis
Data pipeline & experimental skeleton code + on-the-fly mel-spectrogram augmentation
------------------------------------------------
usage: milestone1_adjusted.py [options]
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import librosa
import warnings
# Suppress warnings when n_fft is larger than input signal length
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
from torch.utils.data import DataLoader, Dataset

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
FMAX = 4_000               # adapt to 8 kHz original audio
N_FFT = 512                # FFT window size ≤ frame length
MAX_FRAMES = 1_024         # ≈10 s @ 100 fps

RANDOM_SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count() or 2
# ------------------------------------------------

# ---------- Data utilities ----------
def list_wav_files() -> List[Tuple[Path, int]]:
    """Return list of (file_path, label) where label 0 = HC, 1 = PD."""
    wavs: list[tuple[Path, int]] = []
    for label_name, label in [("HC_AH", 0), ("PD_AH", 1)]:
        for wav in (DATA_ROOT / label_name).glob("*.wav"):
            wavs.append((wav, label))
    return wavs


def load_and_preprocess(path: Path) -> np.ndarray:
    """Load wav, compute log-mel spectrogram (T × M), pad/truncate to MAX_FRAMES."""
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
    # pad / truncate
    T, M = logmel.shape
    if T < MAX_FRAMES:
        pad = MAX_FRAMES - T
        logmel = np.pad(logmel, ((0, pad), (0, 0)), mode='constant', constant_values=-80.0)
    else:
        logmel = logmel[:MAX_FRAMES]
    return logmel


def cache_all(plot: bool = False):
    """Precompute mel-spectrograms to disk for speed"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for wav, _ in list_wav_files():
        out = CACHE_DIR / f"{wav.parent.name.split('_')[0]}_{wav.stem}.npy"
        if not out.exists():
            spec = load_and_preprocess(wav)
            np.save(out, spec)
            if plot:
                plot_spectrogram(spec, wav.name)
    print("[cache_all] Done caching mel-spectrograms")


def plot_spectrogram(spec: np.ndarray, name: str):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.T, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Log-mel spectrogram of {name}")
    plt.xlabel("Time (frames)")
    plt.ylabel("Mel bands")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{name}.png")
    plt.close()

# ---------- Augmentation ----------

def spec_augment(mel: np.ndarray,
                 num_time_masks: int = 2,
                 num_freq_masks: int = 2,
                 max_time_mask_frac: float = 0.1,
                 max_freq_mask_frac: float = 0.1,
                 replace_with_mean: bool = True) -> np.ndarray:
    augmented = mel.copy()
    T, M = augmented.shape
    mask_val = augmented.mean() if replace_with_mean else 0.0
    # time masking
    max_t = int(T * max_time_mask_frac)
    for _ in range(num_time_masks):
        t = np.random.randint(0, max_t)
        t0 = np.random.randint(0, T - t)
        augmented[t0:t0+t, :] = mask_val
    # freq masking
    max_f = int(M * max_freq_mask_frac)
    for _ in range(num_freq_masks):
        f = np.random.randint(0, max_f)
        f0 = np.random.randint(0, M - f)
        augmented[:, f0:f0+f] = mask_val
    return augmented


def time_warp(mel: np.ndarray, max_warp: float = 0.2) -> np.ndarray:
    """Randomly stretch/compress along time by a factor in [1-max_warp, 1+max_warp]."""
    T, M = mel.shape
    rate = 1.0 + np.random.uniform(-max_warp, max_warp)
    warped = librosa.effects.time_stretch(mel.T, rate=rate)
    # pad / truncate back to T
    if warped.shape[1] > T:
        warped = warped[:, :T]
    else:
        pad = T - warped.shape[1]
        warped = np.pad(warped, ((0, 0), (0, pad)), mode="constant", constant_values=warped.mean())
    return warped.T


def add_noise(mel: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
    dynamic = mel.max() - mel.min()
    return mel + np.random.randn(*mel.shape) * (noise_level * dynamic)


def time_shift(mel: np.ndarray, max_shift_frac: float = 0.1) -> np.ndarray:
    T, _ = mel.shape
    shift = np.random.randint(-int(T*max_shift_frac), int(T*max_shift_frac))
    return np.roll(mel, shift, axis=0)

# ---------- Dataset ----------

class ParkinsonDataset(Dataset):
    def __init__(self, wav_paths: List[Path], augment: bool = False):
        self.wav_paths = wav_paths
        self.labels = [0 if p.parent.name.startswith('HC_') else 1 for p in wav_paths]
        self.augment = augment

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav = self.wav_paths[idx]
        mel = load_and_preprocess(wav)
        if self.augment:
            mel = time_warp(mel)
            mel = spec_augment(mel)
            mel = add_noise(mel)
            mel = time_shift(mel)
        # ensure float32 for PyTorch
        mel = mel.astype(np.float32)
        return torch.from_numpy(mel), torch.tensor(self.labels[idx], dtype=torch.float32)

# ---------- Model ----------

class LSTMAudioClassifier(nn.Module):
    def __init__(self, n_mels: int = N_MELS, hidden: int = 128, layers: int = 1,
                 dropout: float = 0.3, bidir: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_mels, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=dropout, bidirectional=bidir)
        feat = hidden * (2 if bidir else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat), nn.Dropout(dropout),
            nn.Linear(feat, feat//4), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(feat//4, 1)
        )

    def forward(self, x):
        out,_ = self.lstm(x)
        h = out.mean(dim=1)
        return self.classifier(h).squeeze(1)

# ---------- Metrics & plotting helpers ----------

def evaluate(model, loader, device="cpu"):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits_list.append(logits.cpu())
            labels_list.append(y.cpu())
    logits = torch.cat(logits_list).numpy()
    y_true = torch.cat(labels_list).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(np.int32)
    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, preds)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc, "cm": cm, "probs": probs, "y_true": y_true}


def plot_confusion_matrix(cm: np.ndarray, path: Path):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["HC", "PD"])
    plt.yticks(ticks, ["HC", "PD"])
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, probs: np.ndarray, path: Path):
    try:
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        return
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve (AUC = {auc:.3f})")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ---------- Training helpers ----------

def step_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * y.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds.cpu() == y.cpu()).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

# ---------- CLI & Entry Point ----------

def main():
    parser = argparse.ArgumentParser(
        prog="milestone1_adjusted.py",
        description="Early Parkinson's detection with mel-spectrogram augmentation"
    )
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--plot", action="store_true", help="Save spectrogram plots to disk")
    args = parser.parse_args()

    print(f"RUN TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    cache_all(plot=args.plot)

    # prepare data
    wav_label_pairs = list_wav_files()
    wavs, labels = zip(*wav_label_pairs)
    train_wavs, val_wavs, train_lbls, val_lbls = train_test_split(
        wavs, labels, test_size=0.3, stratify=labels, random_state=RANDOM_SEED
    )
    train_dl = DataLoader(ParkinsonDataset(train_wavs, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dl   = DataLoader(ParkinsonDataset(val_wavs, augment=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAudioClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = step_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = step_epoch(model, val_dl, criterion, None, device)
        print(f"Epoch {epoch:02d} | train_acc {tr_acc:.3f} | val_acc {val_acc:.3f} | train_loss {tr_loss:.3f} | val_loss {val_loss:.3f}")
        metrics = evaluate(model, val_dl, device)
        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            ckpt_path = CHECKPOINT_DIR / f"best_auc_{best_auc:.3f}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "metrics": metrics}, ckpt_path)
            print(f"[CHECKPOINT] Saved new best AUC: {best_auc:.3f} at {ckpt_path.name}")

    final_metrics = evaluate(model, val_dl, device)
    print("--- FINAL METRICS ---")
    for key, val in final_metrics.items():
        if key in ["probs", "y_true"]:
            continue
        print(f"{key:10}: {val}")

    plot_confusion_matrix(final_metrics["cm"], STATS_DIR / "confusion_matrix.png")
    plot_roc_curve(final_metrics["y_true"], final_metrics["probs"], STATS_DIR / "roc_curve.png")
    print(f"Artifacts saved in {STATS_DIR}")

if __name__ == "__main__":
    main()

