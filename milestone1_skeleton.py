#!/usr/bin/env python
"""
Milestone 1 - Early Parkinson's Detection Using Speech Analysis
Data pipeline & experimental skeleton code + **metrics & checkpoints**
------------------------------------------------
This script pre-computes log-mel spectrograms from the raw wav files and
runs a minimal LSTM classifier. It now also:
  • Saves the **best checkpoint** (highest val ROC-AUC).
  • Computes **accuracy, precision, recall, F1, ROC-AUC** at the end of training.
  • Exports **confusion-matrix** and **ROC curve** plots under `artifacts/plots/`.

usage: milestone1_skeleton.py [options]

Early Parkinson's Detection Using Speech Analysis - Milestone 1

options:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train the model (default: 10)
  -c COMMENTS, --comments COMMENTS
                        Comments to be printed in the log (default: None)
  --plot                Compute and save the spectrograms (default: False)

Example: python milestone1_skeleton.py # quick 10-epoch smoke test 
Example: python milestone1_skeleton.py -e 50 --plot # longer run with plotting

Folder layout expected
----------------------
project-root/
 ├─ data-source/audio/HC_AH/   41 wav (healthy)
 │                   /PD_AH/   40 wav (Parkinson's)
 ├─ artifacts/mel_specs/   (auto-generated)
 ├─ artifacts/plots/       (auto-generated)
 ├─ artifacts/checkpoints/ (auto-generated)
 ├─ artifacts/stats/       (auto-generated)
 └─ milestone1_skeleton.py (this file)
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import librosa
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

SAMPLE_RATE = 16_000        # 16 kHz
N_MELS = 64
HOP_LENGTH = 160            # 10 ms
WIN_LENGTH = 400            # 25 ms
FMIN = 50
FMAX = 4_000                # adapt to 8 kHz originals (Nyquist = SampleRate / 2)
MAX_FRAMES = 1_024          # ≈10 s @ 100 fps

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

    # Pad / truncate to MAX_FRAMES
    n_frames = logmel.shape[0]
    if n_frames >= MAX_FRAMES:
        logmel = logmel[:MAX_FRAMES]
    else:
        pad = MAX_FRAMES - n_frames
        logmel = np.pad(logmel, ((0, pad), (0, 0)), mode="constant", constant_values=-80.0)
    return logmel


def cache_all(plot: bool = False):
    """Pre-compute spectrograms once; safe to skip on subsequent runs."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for wav, _ in list_wav_files():
        label_prefix = wav.parent.name.split("_")[0]  # "HC" or "PD"
        out = CACHE_DIR / f"{label_prefix}_{wav.stem}.npy"
        if not out.exists():
            spec = load_and_preprocess(wav)
            np.save(out, spec)
            if plot:
                plot_spectrogram(spec, wav, label_prefix)
    print("[cache_all] Spectrogram caching DONE")


def plot_spectrogram(spec: np.ndarray, wav: Path, label_prefix: str):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.T, aspect="auto", origin="lower")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Log-mel spectrogram of {wav.name} ({label_prefix})")
    plt.xlabel("Time (frames)")
    plt.ylabel("Mel bands")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{label_prefix}_{wav.stem}.png")
    plt.close()


class ParkinsonDataset(Dataset):
    def __init__(self, files: List[Path]):
        self.files = files
        self.labels = [0 if f.name.startswith("HC_") else 1 for f in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spec = np.load(self.files[idx])  # (T, M)
        return (
            torch.from_numpy(spec),  # float32
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# ---------- Model ----------

class LSTMAudioClassifier(nn.Module):
    def __init__(
        self,
        n_mels: int = N_MELS,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.5,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        d = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, d // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d // 4, 1),
        )

    def forward(self, x):            # x: (B, T, M)
        out, _ = self.lstm(x)        # out: (B, T, d)
        # try mean‐pooling instead of last‐step
        h = out.mean(dim=1)          # (B, d)
        logits = self.classifier(h)  # (B, 1)
        return logits.squeeze(1)


# ---------- Metrics & plots ----------

def evaluate(model, loader: DataLoader, device="cpu"):
    model.eval()
    logits_lst, labels_lst = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits_lst.append(logits.cpu())
            labels_lst.append(y.cpu())
    logits = torch.cat(logits_lst).numpy()
    y_true = torch.cat(labels_lst).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(np.float32)

    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, probs)
    except ValueError:  # happens when only one class present
        roc_auc = float("nan")
    cm = confusion_matrix(y_true, preds)
    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "cm": cm,
        "probs": probs,
        "y_true": y_true,
    }


def plot_confusion_matrix(cm: np.ndarray, path: Path):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["HC", "PD"])
    plt.yticks(tick_marks, ["HC", "PD"])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, probs: np.ndarray, path: Path):
    try:
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        return  # skip if ROC undefined
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
        epilog="Example: python milestone1_skeleton.py                 # quick 10-epoch smoke test"
            "Example: python milestone1_skeleton.py -e 50 --plot    # longer run with plotting",
    )
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-c", "--comments", type=str, default=None, help="Optional log comment")
    parser.add_argument("--plot", action="store_true", default=False, help="Save individual spectrograms")
    args = parser.parse_args()

    print(f"EXECUTION TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.comments:
        print("[COMMENT]", args.comments)

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
        num_workers=NUM_WORKERS
    )
    val_dl = DataLoader(
        ParkinsonDataset(val_files),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAudioClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    best_roc = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = step_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = step_epoch(model, val_dl, criterion, None, device)
        print(
            f"Epoch {epoch:02d} | train acc {tr_acc:.3f} | val acc {val_acc:.3f} | "
            f"train loss {tr_loss:.3f} | val loss {val_loss:.3f}"
        )

        # Checkpoint if ROC improves
        val_metrics = evaluate(model, val_dl, device)
        if not np.isnan(val_metrics["roc_auc"]) and val_metrics["roc_auc"] > best_roc:
            best_roc = val_metrics["roc_auc"]
            ckpt_path = CHECKPOINT_DIR / f"best_auc_{best_roc:.3f}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": val_metrics,
            }, ckpt_path)
            print(f"[CHECKPOINT] Saved new best model → {ckpt_path.name}")

    # ----- Final eval & plots -----
    final_metrics = evaluate(model, val_dl, device)
    print("--- FINAL VALIDATION METRICS ---")
    for k, v in final_metrics.items():
        if not isinstance(v, np.ndarray):
            print(f"{k:10}: {v}")

    # Plots
    plot_confusion_matrix(final_metrics["cm"], STATS_DIR / "confusion_matrix.png")
    plot_roc_curve(final_metrics["y_true"], final_metrics["probs"], STATS_DIR / "roc_curve.png")
    print("Confusion matrix and ROC curve saved to artifacts/plots/")

    print("[INFO] Finished\n")


if __name__ == "__main__":
    main()
