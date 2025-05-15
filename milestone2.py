#!/usr/bin/env python
"""
Milestone 1 – Early Parkinson's Detection Using Speech Analysis
**Windowed data pipeline + LSTM baseline**
--------------------------------------------------------------
*This version fixes the empty‑dataset crash (num_samples=0) by*

1. **Always generating overlapping 2‑s windows** with `cache_all_windows`.
2. Adding clear sanity checks that print how many *.npy* files were found
   and how many belong to each dataset split.
3. Aborting early (with a helpful message) if no cached files are found,
   so you never hit the RandomSampler error again.

Usage examples
```
python milestone1_windowed.py            # quick 10‑epoch smoke test
python milestone1_windowed.py -e 50 --plot   # longer run + plots
```
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
DATA_ROOT       = Path("data-source/audio")
CACHE_DIR       = Path("artifacts/mel_specs")
PLOT_DIR        = Path("artifacts/plots")
CHECKPOINT_DIR  = Path("artifacts/checkpoints")
STATS_DIR       = Path("artifacts/stats")

SAMPLE_RATE = 16_000  # 16 kHz
N_MELS      = 64
HOP_LENGTH  = 160      # 10 ms → 100 frames/s
WIN_LENGTH  = 400      # 25 ms
FMIN, FMAX  = 50, 4_000

# ---- Window params ----
WINDOW_SEC   = 2.0
FRAMES_PER_S = SAMPLE_RATE // HOP_LENGTH  # 100
WIN_FRAMES   = int(WINDOW_SEC * FRAMES_PER_S)  # 200 frames / window
HOP_FRAMES   = WIN_FRAMES // 2  # 50 % overlap (1 s)
PAD_VALUE_DB = -80.0
# -------------------------------------------------

RANDOM_SEED = 42
BATCH_SIZE  = 8
NUM_WORKERS = os.cpu_count() or 2

# ---------- Utility helpers ----------

def list_wav_files() -> List[Tuple[Path, int]]:
    """Return list of (file_path, label) where label 0 = HC, 1 = PD."""
    out: list[tuple[Path, int]] = []
    for label_name, label in [("HC_AH", 0), ("PD_AH", 1)]:
        for wav in (DATA_ROOT / label_name).glob("*.wav"):
            out.append((wav, label))
    return out


def plot_spectrogram(spec: np.ndarray, wav: Path, label_prefix: str):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.T, aspect="auto", origin="lower")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Log‑mel spectrogram – {wav.name} ({label_prefix})")
    plt.xlabel("Time (frames)")
    plt.ylabel("Mel bands")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{label_prefix}_{wav.stem}.png")
    plt.close()


def cache_all_windows(plot: bool = False):
    """Slice each recording into 2‑s windows (50 % overlap) and cache them."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR .mkdir(parents=True, exist_ok=True)

    for wav, _ in list_wav_files():
        y, _ = librosa.load(wav, sr=SAMPLE_RATE)
        y = librosa.util.normalize(y)

        melspec = librosa.feature.melspectrogram(
            y, sr=SAMPLE_RATE, n_mels=N_MELS,
            hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
            fmin=FMIN, fmax=FMAX, power=2.0,
        )
        full_db = librosa.power_to_db(melspec, ref=np.max).T.astype(np.float32)
        total_frames = full_db.shape[0]

        for start in range(0, total_frames, HOP_FRAMES):
            end    = start + WIN_FRAMES
            window = full_db[start:end]

            if window.shape[0] < WIN_FRAMES:  # pad *last* window
                pad = WIN_FRAMES - window.shape[0]
                window = np.pad(window, ((0, pad), (0, 0)),
                                 mode="constant", constant_values=PAD_VALUE_DB)

            win_idx   = start // HOP_FRAMES
            label_pf  = wav.parent.name.split("_", 1)[0]  # HC / PD
            out_name  = f"{label_pf}_{wav.stem}_win{win_idx:03d}.npy"
            np.save(CACHE_DIR / out_name, window)

            if plot and win_idx == 0:
                plot_spectrogram(window, wav, label_pf)
    print("[cache_all_windows] DONE – cached 2‑s windows under", CACHE_DIR)

# ---------- Dataset ----------
class ParkinsonDataset(Dataset):
    """Window‑level dataset. Each item → (T=200, M=64)."""
    def __init__(self, files: List[Path]):
        self.files  = files
        self.labels = [0 if f.name.startswith("HC_") else 1 for f in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spec = np.load(self.files[idx])  # (T, M)
        return torch.from_numpy(spec), torch.tensor(self.labels[idx], dtype=torch.float32)

# ---------- Model ----------
class LSTMAudioClassifier(nn.Module):
    def __init__(self, n_mels=N_MELS, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_mels, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.out  = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

    def forward(self, x):  # x: (B, T, M)
        out, _ = self.lstm(x)
        return self.out(out[:, -1, :]).squeeze(1)

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
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(y_true, probs)
    except ValueError:
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



# ---------- Helper: speaker‑level split ----------

def make_splits():
    all_npys = sorted(CACHE_DIR.glob("*.npy"))
    if not all_npys:
        raise RuntimeError("No .npy files found – run with --recache or check paths!")

    stems = {p.stem.rsplit("_win", 1)[0] for p in all_npys}
    labels_per_stem = [0 if s.startswith("HC_") else 1 for s in stems]

    train_stems, val_stems = train_test_split(
        list(stems), test_size=0.3, stratify=labels_per_stem, random_state=RANDOM_SEED
    )

    def stems_to_windows(stem_set):
        return [p for p in all_npys if p.stem.rsplit("_win", 1)[0] in stem_set]

    return stems_to_windows(train_stems), stems_to_windows(val_stems)


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

# ---------- Main ----------

def main():
    argp = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argp.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    argp.add_argument("--plot", action="store_true", help="Save one spectrogram per speaker")
    argp.add_argument("--recache", action="store_true", help="Force re‑compute windows")
    args = argp.parse_args()

    print("EXECUTION TIME:", datetime.now().strftime("%Y‑%m‑%d %H:%M:%S"))

    if args.recache or not any(CACHE_DIR.glob("*.npy")):
        cache_all_windows(args.plot)
    else:
        print("[info] Found cached windows – skipping preprocessing.")

    train_files, val_files = make_splits()
    print(f"Train windows: {len(train_files)},  Val windows: {len(val_files)}")

    if len(train_files) == 0:
        raise RuntimeError("Zero train examples – check CACHE_DIR content and file naming!")

    train_dl = DataLoader(ParkinsonDataset(train_files), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS)
    val_dl   = DataLoader(ParkinsonDataset(val_files), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMAudioClassifier().to(device)
    criterion, optimizer = nn.BCEWithLogitsLoss(), torch.optim.Adam(model.parameters(), lr=1e-3)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR     .mkdir(parents=True, exist_ok=True)
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
    plot_confusion_matrix(final_metrics["cm"], PLOT_DIR / "confusion_matrix.png")
    plot_roc_curve(final_metrics["y_true"], final_metrics["probs"], PLOT_DIR / "roc_curve.png")
    print("Confusion matrix and ROC curve saved to artifacts/plots/")

    print("[INFO] Finished\n")

if __name__ == "__main__":
    main()
