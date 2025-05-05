#!/usr/bin/env python
"""
Milestone 1 - Early Parkinson's Detection Using Speech Analysis
Data pipeline & experimental skeleton code
Group: Antonio Alampi (64316), Tommaso Tragno (64699), Cristian Tedesco (65149), Pol Rion Solé (65177)

This script pre-computes log-mel spectrograms from the raw wav files and
provides a minimal PyTorch training skeleton with an LSTM classifier.
It will be possible to edit the hyper-parameters in the CONFIG section to explore variations,
or swap out the model in **LSTMAudioClassifier** with more advanced architectures in later milestones.

Usage
-----
$ python milestone1_skeleton.py                 # first run - caches spectrograms
Arguments
-----
    --epochs <int>   Number of epochs to train the model (default: 10)
    --comments <str> Comments to be printed in the log (default: None)
    --plot           Compute and save the spectrograms (default: False)
    --help           Show this help message and exit

Folder layout expected
----------------------
project-root/
 ├─ data-source/
 │   └─ audio/
 │      ├─ HC_AH/   (Healthy Control - 41 wav)
 │      └─ PD_AH/   (Parkinson's Disease - 40 wav)
 └─ milestone1_skeleton.py

(NOTE: every wav filename starts with **AH** irrespective of class, so we embed the class label in the *cached* filename instead.)

Outputs
-------
artifacts/mel_specs/HC_*.npy or PD_*.npy - fixed-length (MAX_FRAMES, N_MELS)
artifacts/plots/*.png                  - spectrogram plots (for debugging)
artifacts/checkpoints/*.pt             - model weights (TODO)
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime
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
SAMPLE_RATE = 16_000      # 16kHz
N_MELS = 64
HOP_LENGTH = 160          # 10 ms
WIN_LENGTH = 400          # 25 ms
FMIN = 50
# FMAX = SAMPLE_RATE // 2  # Nyquist frequency
# Since the original data is sampled at 8kHz, we should (?) set FMAX to a lower value
FMAX = 4_000 # 8_000
MAX_FRAMES = 1024          # ≈10 s at 100 fps
RANDOM_SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count() or 2
# ------------------------------------------------


def list_wav_files() -> List[Tuple[Path, int]]:
    """Return list of (file_path, label) where label 0 = HC, 1 = PD."""
    wavs: list[tuple[Path, int]] = []
    for label_name, label in [("HC_AH", 0), ("PD_AH", 1)]:
        for wav in (DATA_ROOT / label_name).glob("*.wav"):
            wavs.append((wav, label))
    return wavs


def load_and_preprocess(path: Path) -> np.ndarray:
    """Load wav and compute a padded / truncated log-mel spectrogram (T × M)."""
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
        # Pad with -80 dB (minimum value in librosa, which mean silence) instead of 0
        # to avoid introducing artifacts in the spectrogram
        logmel = np.pad(logmel, ((0, pad), (0, 0)), mode="constant", constant_values=-80.0)
    return logmel


def cache_all(plot: bool = False):
    """Pre-compute spectrograms once; safe to skip on subsequent runs."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for wav, _ in list_wav_files():
        # Embed the class label (HC/PD) in the cached filename because the raw filename is class-agnostic (starts with AH)
        label_prefix = wav.parent.name.split("_")[0]  # "HC" or "PD"
        out = CACHE_DIR / f"{label_prefix}_{wav.stem}.npy"
        if not out.exists():
            spec = load_and_preprocess(wav)
            np.save(out, spec)
            if plot:
                PLOT_DIR.mkdir(parents=True, exist_ok=True)
                plot_spectrogram(spec, wav, label_prefix)
    print("[cache_all] Spectrogram caching DONE")


def plot_spectrogram(spec: np.ndarray, wav: Path, label_prefix: str):
    """Plot and save the spectrogram for debugging."""
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.T, aspect="auto", origin="lower")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Log-mel spectrogram of {wav.name} ({label_prefix})")
    plt.xlabel("Time (frames)")
    plt.ylabel("Mel bands")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{label_prefix}_{wav.stem}.png")
    plt.clf()
    plt.close()


class ParkinsonDataset(Dataset):
    def __init__(self, files: List[Path]):
        self.files = files
        # Label by filename prefix (HC_ vs PD_) injected during caching
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
    def __init__(self, n_mels: int = N_MELS, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
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
    parser = argparse.ArgumentParser(
        prog="milestone1_skeleton.py",
        usage="%(prog)s [options]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Early Parkinson's Detection Using Speech Analysis - Milestone 1",
        epilog="Example: python milestone1_skeleton.py --epochs 20 --comments 'First run' --plot"
    )
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("-c", "--comments", type=str, default=None, help="Comments to be printed in the log")
    parser.add_argument("--plot", action="store_true", default=False, help="Compute and save the spectrograms")
    args = parser.parse_args()

    print(f"EXECUTION TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(args.comments)

    cache_all(args.plot)

    all_files = sorted(CACHE_DIR.glob("*.npy"))
    labels = [0 if f.name.startswith("HC_") else 1 for f in all_files]
    train_files, val_files = train_test_split(
        all_files,
        test_size=0.3,
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
        print(f"Epoch {epoch:02d} | train acc {tr_acc:.3f} | val acc {val_acc:.3f} | train loss {tr_loss:.3f} | val loss {val_loss:.3f}")

    # TODO: save checkpoints, metrics, confusion matrix, ROC-AUC

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
import joblib

# === Checkpoint saving ===
def save_checkpoint(model, optimizer, epoch, loss, path='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

# === Metrics & Confusion Matrix ===
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).int()
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Save metrics
    auc_score = roc_auc_score(all_labels, all_preds)
    print(f"ROC-AUC: {auc_score:.4f}")

    # Save confusion matrix
    cm = confusion_matrix(all_labels, (np.array(all_preds) > 0.5).astype(int))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion matrix as confusion_matrix.png")

    # Save ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("roc_curve.png")
    print("Saved ROC curve as roc_curve.png")

    return auc_score

if __name__ == "__main__":
    main()
    print("[INFO] Finished\n\n")
