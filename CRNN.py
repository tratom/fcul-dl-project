#!/usr/bin/env python3
"""
Early Parkinson's Detection Using Speech Analysis - Milestone 2
Cleaned version: no augmentation generation or specaugment.
"""
from __future__ import annotations
import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

# -------------------- CONFIG --------------------

DATA_ROOT      = Path("data_original+aug") # Path("data-source/audio")
AUG_DIR        = Path("artifacts/augmented_audio")
PLOT_DIR       = Path("milestone2/plots_augmented")
CHECKPOINT_DIR = Path("milestone2/checkpoints_augmented")
STATS_DIR      = Path("milestone2/stats_augmented")
for d in (CHECKPOINT_DIR, PLOT_DIR, STATS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16_000
N_MELS      = 64
HOP_LENGTH  = 160
WIN_LENGTH  = 400
FMIN        = 50
FMAX        = 4_000
MAX_FRAMES  = 1_024
SPEC_PAD_VALUE = -80.0

RANDOM_SEED = 42
BATCH_SIZE  = 8
NUM_WORKERS = os.cpu_count() or 2
EPSILON     = 0.1

# -------------------- PREPROCESSING --------------------
def load_and_preprocess(
    wav_path: Path,
    spec_path: Path,
    mask_path: Path,
    plot_path: Path | None,
    do_plot: bool
) -> None:
    if spec_path.exists() and mask_path.exists() and (not do_plot or (plot_path and plot_path.exists())):
        return
    y, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE)
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y, top_db=35)
    melspec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmin=FMIN, fmax=FMAX,
        power=2.0
    )
    logmel = librosa.power_to_db(melspec, ref=np.max).astype(np.float32)
    delta  = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)
    full   = np.stack([logmel, delta, delta2], axis=0)
    T = full.shape[2]
    if T >= MAX_FRAMES:
        full = full[:, :, :MAX_FRAMES]
        mask = np.ones(MAX_FRAMES, dtype=np.uint8)
    else:
        pad = MAX_FRAMES - T
        full = np.pad(full, ((0,0),(0,0),(0,pad)), constant_values=SPEC_PAD_VALUE)
        mask = np.concatenate([np.ones(T, dtype=np.uint8), np.zeros(pad, dtype=np.uint8)])
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(spec_path, full)
    np.save(mask_path, mask)
    if do_plot and plot_path:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        if not plot_path.exists():
            plt.figure(figsize=(10,4))
            librosa.display.specshow(
                logmel, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                x_axis='time', y_axis='mel', fmin=FMIN, fmax=FMAX
            )
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

class MelSpecDataset(Dataset):
    def __init__(self, files: List[Path]):
        self.files = files
        self.labels = [0 if f.parent.name=='HC_AH' else 1 for f in files]
    def __len__(self) -> int:
        return len(self.files)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spec = np.load(str(self.files[idx]))              # (3, N_MELS, MAX_FRAMES)
        mask = np.load(str(self.files[idx].with_suffix('.mask.npy')))
        x = torch.from_numpy(spec)                        # float32
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

# -------------------- MODEL --------------------
class CRNNClassifier(nn.Module):
    def __init__(self, n_mels: int = N_MELS, hidden_size: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2,2))
        )
        self.lstm = nn.LSTM(
            input_size=64 * (n_mels // 8),
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.permute(0, 3, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        B, T, C, F = x.shape
        x = x.reshape(B, T, C * F)
        out, _ = self.lstm(x)
        out_max, _ = torch.max(out, dim=1)
        out_avg = torch.mean(out, dim=1)
        out = torch.cat([out_max, out_avg], dim=1)
        out = self.dropout(out)
        return self.classifier(out).squeeze(1)

# -------------------- METRICS --------------------
def evaluate(model: nn.Module, loader: DataLoader, device='cpu') -> dict:
    preds, labels, probs = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)
            probs.append(prob.cpu().numpy())
            labels.append(y.cpu().numpy())
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probs)

    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]

    y_pred = (y_prob > best_thresh).astype(np.float32)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    return {
        'acc': accuracy_score(y_true, y_pred),
        'precision': prec, 'recall': rec, 'f1': f1,
        'roc_auc': roc_auc_score(y_true, y_prob),
        'cm': confusion_matrix(y_true, y_pred),
        'y_true': y_true, 'probs': y_prob, 'threshold': best_thresh
    }


def plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(4,4)); plt.imshow(cm, cmap='Blues'); plt.title('Confusion Matrix');
    plt.colorbar(); ticks=['HC','PD']; plt.xticks([0,1],ticks); plt.yticks([0,1],ticks);
    thresh=cm.max()/2
    for i in range(2):
        for j in range(2): plt.text(j,i,cm[i,j],ha='center',va='center',
                                     color='white' if cm[i,j]>thresh else 'black')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout(); plt.savefig(path); plt.close()

def plot_roc_curve(y_true: np.ndarray, probs: np.ndarray, path: Path) -> None:
    try:
        fpr,tpr,_ = roc_curve(y_true,probs)
        auc = roc_auc_score(y_true,probs)
    except ValueError:
        return
    plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],linestyle='--');
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate');
    plt.title(f'ROC Curve (AUC={auc:.3f})'); plt.tight_layout(); plt.savefig(path); plt.close()

# -------------------- TRAINING HELPER --------------------
def step_epoch(
    model: nn.Module, loader: DataLoader,
    criterion: nn.Module, optimizer: torch.optim.Optimizer|None,
    device: str='cpu', epsilon: float=0.0
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, samples = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_train: optimizer.zero_grad()
        y_s = y * (1 - epsilon) + 0.5 * epsilon
        logits = model(x)
        loss = criterion(logits, y_s)
        if is_train:
            loss.backward()
            optimizer.step()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
        samples += y.size(0)
        total_loss += loss.item() * y.size(0)
    return total_loss / samples, correct / samples

# -------------------- MAIN --------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--epochs',type=int,default=10)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    print(f"EXECUTION TIME: {datetime.now():%Y-%m-%d %H:%M:%S}")

    # 1) PREPROCESS (now including test)
    for split in ('training','validation','test'):
        for lbl in ('HC_AH','PD_AH'):
            wav_dir = DATA_ROOT/split/lbl
            plot_base = PLOT_DIR/split/lbl if args.plot else None
            for wav in sorted(wav_dir.glob('*.wav')):
                spec = wav.with_suffix('.npy')
                mask = wav.with_suffix('.mask.npy')
                plot = plot_base/f"{wav.stem}.png" if plot_base else None
                load_and_preprocess(wav, spec, mask, plot, args.plot)

    # LOAD training and validation files
    train_files: List[Path] = []
    for lbl in ('HC_AH','PD_AH'):
        train_files.extend(sorted((DATA_ROOT/'training'/lbl).glob('*.npy')))
    train_files = [f for f in train_files if not f.name.endswith('.mask.npy')]

    val_files: List[Path] = []
    for lbl in ('HC_AH','PD_AH'):
        val_files.extend(sorted((DATA_ROOT/'validation'/lbl).glob('*.npy')))
    val_files = [f for f in val_files if not f.name.endswith('.mask.npy')]

    test_files = []
    for lbl in ('HC_AH','PD_AH'):
        test_files += sorted((DATA_ROOT/'test'/lbl).glob('*.npy'))
    test_files = [f for f in test_files if not f.name.endswith('.mask.npy')]

    train_dl = DataLoader(MelSpecDataset(train_files), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS)
    val_dl   = DataLoader(MelSpecDataset(val_files),   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)
    test_dl  = DataLoader(MelSpecDataset(test_files), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNNClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_auc = -np.inf
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = step_epoch(model, train_dl, criterion, optimizer, device, EPSILON)
        val_loss, val_acc = step_epoch(model, val_dl, criterion, None, device, 0.0)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d} | train acc {tr_acc:.3f} | val acc {val_acc:.3f} | train loss {tr_loss:.3f} | val loss {val_loss:.3f}")
        metrics = evaluate(model, val_dl, device)
        if not np.isnan(metrics['roc_auc']) and metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            ckpt = CHECKPOINT_DIR / f"best_auc_{best_auc:.3f}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, ckpt)
            print(f"[CHECKPOINT] Saved {ckpt.name}")

    # FINAL EVAL
    final = evaluate(model, val_dl, device)
    print("--- FINAL METRICS ---")
    for k,v in final.items():
        if not isinstance(v, np.ndarray): print(f"{k:10}: {v}")
    plot_confusion_matrix(final['cm'], STATS_DIR/"confusion_matrix.png")
    plot_roc_curve(final['y_true'], final['probs'], STATS_DIR/"roc_curve.png")

    # 7) TEST RESULTS (load best model)
    best_ckpt = CHECKPOINT_DIR / f"best_auc_{best_auc:.3f}.pt"

    # torch.load in 2.6+ defaults to weights_only=True which rejects numpy scalars.
    # We need weights_only=False so it will load the entire dict, then pick out the state_dict.
    ckpt_data = torch.load(best_ckpt, map_location=device, weights_only=False)

    # if you saved a dict with 'model_state_dict', grab that, otherwise assume it's the raw state_dict
    if isinstance(ckpt_data, dict) and 'model_state_dict' in ckpt_data:
        state_dict = ckpt_data['model_state_dict']
    else:
        state_dict = ckpt_data

    model.load_state_dict(state_dict)
    test_metrics = evaluate(model, test_dl, device)
    print("\n--- TEST METRICS ---")
    for k,v in test_metrics.items():
        if not isinstance(v, np.ndarray):
            print(f"{k:10}: {v:.4f}")
    plot_confusion_matrix(test_metrics['cm'], STATS_DIR/"test_confusion_matrix.png")
    plot_roc_curve(test_metrics['y_true'], test_metrics['probs'], STATS_DIR/"test_roc_curve.png")
    pd.DataFrame({k:[test_metrics[k]] for k in ('acc','precision','recall','f1','roc_auc')}) \
      .to_csv(STATS_DIR/"test_metrics.csv", index=False)

    print("Done.")

# MAIN END

if __name__ == '__main__':
    main()
