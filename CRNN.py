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

DATA_ROOT      = Path("data-source/audio")
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

RANDOM_SEED = 42
BATCH_SIZE  = 8
NUM_WORKERS = os.cpu_count() or 2
EPSILON     = 0.1

# -------------------- PREPROCESSING --------------------
def load_and_preprocess(path: Path) -> np.ndarray:
    if path.suffix == '.npy':
        y = np.load(path)
    else:
        y, _ = librosa.load(path, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y, top_db=25)
    y = librosa.util.normalize(y)
    melspec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        fmin=FMIN, fmax=FMAX, power=2.0,
    )
    logmel = librosa.power_to_db(melspec, ref=np.max).astype(np.float32)
    delta = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)
    full = np.stack([logmel, delta, delta2], axis=0)
    if full.shape[2] >= MAX_FRAMES:
        full = full[:, :, :MAX_FRAMES]
    else:
        pad_w = MAX_FRAMES - full.shape[2]
        full = np.pad(full, ((0,0),(0,0),(0,pad_w)), constant_values=-80.0)
    return full

class MelSpecDataset(Dataset):
    def __init__(self, files: List[Path]):
        self.files = files
        self.labels = [0 if f.parent.name=='HC_AH' else 1 for f in files]
    def __len__(self) -> int:
        return len(self.files)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spec_np = load_and_preprocess(self.files[idx])
        spec_t  = torch.from_numpy(spec_np)
        spec_t = spec_t.permute(1, 2, 0).permute(1, 0, 2)
        label  = torch.tensor(self.labels[idx], dtype=torch.float32)
        return spec_t, label

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
        x = x.permute(0, 3, 2, 1)
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
    args = parser.parse_args()

    print(f"EXECUTION TIME: {datetime.now():%Y-%m-%d %H:%M:%S}")

    orig_paths = []
    for lbl in ('HC_AH','PD_AH'):
        orig_paths.extend(sorted((AUG_DIR/lbl).glob('*.wav')))
    labels_orig = [0 if p.parent.name=='HC_AH' else 1 for p in orig_paths]

    fold_metrics = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for fold, (train_idx, val_idx) in enumerate(skf.split(orig_paths, labels_orig)):
        print(f"Fold {fold+1}/5")
        train_orig = [orig_paths[i] for i in train_idx]
        val_orig = [orig_paths[i] for i in val_idx]

        train_files = list(train_orig)
        for orig in train_orig:
            label = orig.parent.name
            train_files.extend(sorted((AUG_DIR/label).glob(f"{orig.stem}*.npy")))
        val_files = list(val_orig)

        train_dl = DataLoader(
            MelSpecDataset(train_files),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        val_dl = DataLoader(
            MelSpecDataset(val_files),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CRNNClassifier().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        best_auc = -1.0
        for epoch in range(1, args.epochs+1):
            tr_loss, tr_acc = step_epoch(model, train_dl, criterion, optimizer, device, EPSILON)
            val_loss, val_acc = step_epoch(model, val_dl, criterion, None, device, 0.0)
            scheduler.step(val_loss)
            print(f"Epoch {epoch:02d} | train acc {tr_acc:.3f} | val acc {val_acc:.3f} | train loss {tr_loss:.3f} | val loss {val_loss:.3f}")
            metrics = evaluate(model, val_dl, device)
            if not np.isnan(metrics['roc_auc']) and metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                ckpt = CHECKPOINT_DIR / f"fold{fold+1}_best_auc_{best_auc:.3f}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics
                }, ckpt)
                print(f"[CHECKPOINT] Saved {ckpt.name}")

        final = evaluate(model, val_dl, device)
        print("--- FINAL METRICS ---")
        for k,v in final.items():
            if not isinstance(v, np.ndarray): print(f"{k:10}: {v}")
        plot_confusion_matrix(final['cm'], STATS_DIR/f'fold{fold+1}_confusion_matrix.png')
        plot_roc_curve(final['y_true'], final['probs'], STATS_DIR/f'fold{fold+1}_roc_curve.png')
        print("Plots saved for fold.")

        fold_metrics.append(final)

    print("===== AVERAGE METRICS ACROSS FOLDS =====")
    keys = [k for k in fold_metrics[0].keys() if isinstance(fold_metrics[0][k], (int, float))]
    avg = {k: np.mean([m[k] for m in fold_metrics]) for k in keys}
    for k, v in avg.items():
        print(f"{k:10}: {v:.4f}")

    df = pd.DataFrame([{k: v for k, v in m.items() if k in keys} for m in fold_metrics])
    df.loc['mean'] = df.mean(numeric_only=True)
    df.to_csv(STATS_DIR / 'cv_metrics_summary.csv', index_label='fold')

    plt.figure(figsize=(8, 4))
    df.mean(numeric_only=True).plot(kind='bar', yerr=df.std(numeric_only=True), capsize=4)
    plt.title('Average Metrics Across Folds')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(STATS_DIR / 'cv_metrics_barplot.png')
    plt.close()

    all_y_true = np.concatenate([m['y_true'] for m in fold_metrics if 'y_true' in m])
    all_y_prob = np.concatenate([m['probs'] for m in fold_metrics if 'probs' in m])
    fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
    auc = roc_auc_score(all_y_true, all_y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Overall ROC Curve Across All Folds')
    plt.legend()
    plt.tight_layout()
    plt.savefig(STATS_DIR / 'cv_avg_roc_curve.png')
    plt.close()

# ✅ MAIN END

if __name__ == '__main__':
    main()

