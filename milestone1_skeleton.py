#!/usr/bin/env python
"""
Milestone 1 - Early Parkinson's Detection Using Speech Analysis
Data pipeline & experimental skeleton code
Group: Antonio Alampi (64316), Tommaso Tragno (64699), Cristian Tedesco (65149), Pol Rion Solé (65177)

This script pre-computes log-mel spectrograms from the raw wav files and
provides a minimal PyTorch training skeleton with an LSTM classifier.
Edit the hyper-parameters in the CONFIG section to explore variations,
or swap out the model in LSTMAudioClassifier with more advanced
architectures in later milestones.

Usage
-----
$ python milestone1_skeleton.py            # first run - caches spectrograms
$ python milestone1_skeleton.py --epochs 50  # quick experiment

Folder layout expected
----------------------
project-root/
 ├─ data-source/
 │   └─ audio/
 │      ├─ HC_AH/   (41 wav)
 │      └─ PD_AH/   (40 wav)
 └─ milestone1_skeleton.py

Outputs
-------
artifacts/mel_specs/*.npy         - fixed-length (MAX_FRAMES, N_MELS) arrays
artifacts/plots/*.png             - spectrogram plots (for debugging)
artifacts/checkpoints/*.pt        - model weights (TODO)

"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# -------------------- CONFIG --------------------
DATA_ROOT = Path("data-source/audio")
CACHE_DIR = Path("artifacts/mel_specs")
PLOT_DIR = Path("artifacts/plots")
SAMPLE_RATE = 16_000
N_MELS = 64
HOP_LENGTH = 160          # 10 ms
WIN_LENGTH = 400          # 25 ms
FMIN = 50
FMAX = 8_000
MAX_FRAMES = 1_024        # ~10 s at 100 fps
RANDOM_SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count() or 2
# ------------------------------------------------


def list_wav_files() -> List[Tuple[Path, int]]:
    """Return list of (file_path, label) where label 0 = HC, 1 = PD."""
    wavs = []
    for label_name, label in [("HC_AH", 0), ("PD_AH", 1)]:
        for wav in (DATA_ROOT / label_name).glob("*.wav"):
            wavs.append((wav, label))
    return wavs


def load_and_preprocess(path: Path) -> np.ndarray:
    """Load wav, compute log-mel spectrogram (time × n_mels)."""
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    y = librosa.util.normalize(y)
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    logmel = librosa.power_to_db(melspec, ref=np.max).T.astype(np.float32)  # (T, M)

    # Pad / truncate to MAX_FRAMES for batching
    n_frames = logmel.shape[0]
    if n_frames >= MAX_FRAMES:
        logmel = logmel[:MAX_FRAMES]
    else:
        pad = MAX_FRAMES - n_frames
        logmel = np.pad(logmel, ((0, pad), (0, 0)), mode="constant")
    return logmel


def cache_all(plot: bool = False):
    """Pre-compute spectrograms once; safe to skip on subsequent runs."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for wav, _ in list_wav_files():
        out = CACHE_DIR / f"{wav.stem}.npy"
        if not out.exists():
            spec = load_and_preprocess(wav)
            np.save(out, spec)
            if plot:
                PLOT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure plot directory exists
                plot_spectrogram(spec, wav)
    print("[cache_all] Spectrogram caching DONE")

def plot_spectrogram(spec: np.ndarray, wav: Path):
    """Plot and save the spectrogram for debugging."""
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.T, aspect="auto", origin="lower")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Log-mel spectrogram of {wav.name}")
    plt.xlabel("Time (frames)")
    plt.ylabel("Mel bands")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{wav.stem}.png")
    plt.clf()
    plt.close()


class ParkinsonDataset(Dataset):
    def __init__(self, files: List[Path]):
        self.files = files
        # Infer label by filename prefix (HC_ vs PD_)
        self.labels = [0 if f.name.startswith("HC_") else 1 for f in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spec = np.load(self.files[idx])  # (T, M)
        return (
            torch.from_numpy(spec),  # float32
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class LSTMAudioClassifier(nn.Module):
    def __init__(self, n_mels=N_MELS, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):  # x: (B, T, M)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.out(last).squeeze(1)


# ---------- Training helpers ----------

def step_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train() if train else model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * y.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    cache_all(plot=True)

    all_files = sorted(CACHE_DIR.glob("*.npy"))
    labels = [0 if f.name.startswith("HC_") else 1 for f in all_files]
    train_files, val_files = train_test_split(
        all_files,
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    train_dl = DataLoader(
        ParkinsonDataset(train_files),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_dl = DataLoader(
        ParkinsonDataset(val_files),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAudioClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = step_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = step_epoch(model, val_dl, criterion, None, device)
        print(f"Epoch {epoch:02d} | train acc {tr_acc:.3f} | val acc {val_acc:.3f}")

    # TODO: save checkpoints, metrics, confusion matrix, ROC


if __name__ == "__main__":
    main()
