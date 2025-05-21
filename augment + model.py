#!/usr/bin/env python3
"""
Early Parkinson's Detection Using Speech Analysis - Milestone 2
Integrates:
 1) Offline waveform augmentations on training set only (with pink noise)
 2) On-the-fly SpecAugment on log-mel spectrograms
 3) Manual label smoothing in BCEWithLogitsLoss
"""
from __future__ import annotations
import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
DATA_ROOT      = Path("data-source/audio")
AUG_DIR        = Path("artifacts/augmented_audio")
PLOT_DIR       = Path("milestone2/plots_augmented")
CHECKPOINT_DIR = Path("milestone2/checkpoints_augmented")
STATS_DIR      = Path("milestone2/stats_augmented")

SAMPLE_RATE = 16_000   # 16 kHz
N_MELS      = 64
HOP_LENGTH  = 160      # 10 ms
WIN_LENGTH  = 400      # 25 ms
FMIN        = 50
FMAX        = 4_000    # adapt to 8 kHz originals
MAX_FRAMES  = 1_024    # ≈10 s @ 100 fps

RANDOM_SEED = 42
BATCH_SIZE  = 8
NUM_WORKERS = os.cpu_count() or 2
EPSILON     = 0.1      # label smoothing factor

# -------------------- UTIL: PINK NOISE --------------------
def generate_pink_noise(n_samples: int) -> np.ndarray:
    """
    Genera rumore rosa (1/f) con algoritmo di Paul Kellet.
    """
    b0 = b1 = b2 = 0.0
    out = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        white = np.random.randn()
        b0 = 0.99886 * b0 + white * 0.0555179
        b1 = 0.99332 * b1 + white * 0.0750759
        b2 = 0.96900 * b2 + white * 0.1538520
        out[i] = b0 + b1 + b2 + white * 0.5362
    # normalizza a RMS=1
    out = out / np.sqrt(np.mean(out**2))
    return out

# -------------------- 1) OFFLINE AUGMENTATIONS ON TRAIN ONLY --------------------
def generate_offline_augmentations(orig_paths: List[Path], output_dir: Path) -> None:
    """
    Per ogni wav in orig_paths genera 5 augmentazioni e salva come .npy:
      - speed_down  (–5 %)
      - speed_up    (+5 %)
      - pink noise  (20 dB SNR)
      - pitch_down  (–2 semitoni)
      - pitch_up    (+2 semitoni)
    """
    snr_db = 20
    speed_rates = [0.95, 1.05]
    pitch_steps = [-2, 2]
    for wav_path in orig_paths:
        label = wav_path.parent.name  # HC_AH o PD_AH
        out_label_dir = output_dir / label
        out_label_dir.mkdir(parents=True, exist_ok=True)

        # carica segnale originale
        y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        # 1) speed perturb
        for rate, suffix in zip(speed_rates, ["speed_down", "speed_up"]):
            y_sp = librosa.effects.time_stretch(y, rate=rate)
            np.save(out_label_dir / f"{wav_path.stem}_{suffix}.npy", y_sp)

        # 2) pink noise @ 20 dB SNR
        rms_signal = np.sqrt(np.mean(y**2))
        rms_noise  = rms_signal / (10**(snr_db / 20))
        pink = generate_pink_noise(y.shape[0]) * rms_noise
        y_no = y + pink
        np.save(out_label_dir / f"{wav_path.stem}_noise.npy", y_no)

        # 3) pitch shift
        for step, suffix in zip(pitch_steps, ["pitch_down", "pitch_up"]):
            y_ps = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=step)
            np.save(out_label_dir / f"{wav_path.stem}_{suffix}.npy", y_ps)

# -------------------- 2) ON-THE-FLY SPEC-AUGMENT --------------------
class SpecAugment(nn.Module):
    def __init__(
        self,
        max_time_mask: int = 10,
        max_freq_mask: int = 8,
        n_time_masks: int = 1,
        n_freq_masks: int = 1,
    ):
        super().__init__()
        self.max_time_mask = max_time_mask
        self.max_freq_mask = max_freq_mask
        self.n_time_masks  = n_time_masks
        self.n_freq_masks  = n_freq_masks
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        F, T = spec.shape
        spec_aug = spec.clone()
        for _ in range(self.n_time_masks):
            t  = random.randint(1, self.max_time_mask)
            t0 = random.randint(0, T - t)
            spec_aug[:, t0:t0+t] = 0
        for _ in range(self.n_freq_masks):
            f  = random.randint(1, self.max_freq_mask)
            f0 = random.randint(0, F - f)
            spec_aug[f0:f0+f, :] = 0
        return spec_aug

# -------------------- 3) PREPROCESSING --------------------
def load_and_preprocess(path: Path) -> np.ndarray:
    """
    Load wav or npy, return padded/truncated log-mel (F, T).
    """
    if path.suffix == '.npy':
        y = np.load(path)
    else:
        y, _ = librosa.load(path, sr=SAMPLE_RATE)
    y = librosa.util.normalize(y)
    melspec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        fmin=FMIN, fmax=FMAX, power=2.0,
    )
    logmel = librosa.power_to_db(melspec, ref=np.max).astype(np.float32)
    # pad/truncate time axis
    if logmel.shape[1] >= MAX_FRAMES:
        logmel = logmel[:, :MAX_FRAMES]
    else:
        pad_w = MAX_FRAMES - logmel.shape[1]
        logmel = np.pad(logmel, ((0,0),(0,pad_w)), constant_values=-80.0)
    return logmel

class MelSpecDataset(Dataset):
    def __init__(self, files: List[Path], specaugment: SpecAugment|None=None):
        self.files = files
        self.labels = [0 if f.parent.name=='HC_AH' else 1 for f in files]
        self.specaugment = specaugment
    def __len__(self) -> int:
        return len(self.files)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spec_np = load_and_preprocess(self.files[idx])  # (F, T)
        spec_t  = torch.from_numpy(spec_np)
        if self.specaugment:
            spec_t = self.specaugment(spec_t)
        spec_t = spec_t.T
        label  = torch.tensor(self.labels[idx], dtype=torch.float32)
        return spec_t, label

# -------------------- Model --------------------
class LSTMAudioClassifier(nn.Module):
    def __init__(self, n_mels:int=N_MELS,
                 hidden_size:int=128, num_layers:int=2,
                 dropout:float=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mels, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout
        )
        self.out = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size,1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out,_ = self.lstm(x)
        last  = out[:,-1,:]
        return self.out(last).squeeze(1)

# -------------------- Metrics & plots --------------------
def evaluate(model: nn.Module, loader: DataLoader, device:str='cpu') -> dict:
    preds,labels,probs = [],[],[]
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            logits = model(x)
            prob   = torch.sigmoid(logits)
            preds.append((prob>0.5).float().cpu().numpy())
            labels.append(y.cpu().numpy())
            probs.append(prob.cpu().numpy())
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)
    prec,rec,f1,_ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    return {
        'acc':accuracy_score(y_true,y_pred),
        'precision':prec, 'recall':rec, 'f1':f1,
        'roc_auc':roc_auc_score(y_true,y_prob) if len(np.unique(y_true))>1 else float('nan'),
        'cm':confusion_matrix(y_true,y_pred),
        'y_true':y_true, 'probs':y_prob
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

# -------------------- Training helper --------------------
def step_epoch(
    model: nn.Module, loader: DataLoader,
    criterion: nn.Module, optimizer: torch.optim.Optimizer|None,
    device: str='cpu', epsilon: float=0.0
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss,correct,samples = 0.0,0,0
    for x,y in loader:
        x,y = x.to(device),y.to(device)
        if is_train: optimizer.zero_grad()
        y_s = y*(1-epsilon) + 0.5*epsilon
        logits = model(x)
        loss   = criterion(logits,y_s)
        if is_train: loss.backward(); optimizer.step()
        preds  = (torch.sigmoid(logits)>0.5).float()
        correct+= (preds==y).sum().item(); samples+=y.size(0);
        total_loss += loss.item()*y.size(0)
    return total_loss/samples, correct/samples

# -------------------- MAIN & CLI --------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog='milestone2.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Parkinson's Detection via Audio - Milestone 2"
    )
    parser.add_argument('-e','--epochs',type=int,default=10,help='Number of epochs')
    args = parser.parse_args()

    print(f"EXECUTION TIME: {datetime.now():%Y-%m-%d %H:%M:%S}")
    # 1) split originals
    orig_paths: List[Path] = []
    for lbl in ('HC_AH','PD_AH'):
        orig_paths.extend(sorted((DATA_ROOT/lbl).glob('*.wav')))
    labels_orig = [0 if p.parent.name=='HC_AH' else 1 for p in orig_paths]
    train_orig,val_orig = train_test_split(
        orig_paths,test_size=0.3,stratify=labels_orig,random_state=RANDOM_SEED
    )
    # 2) augment only train originals
    generate_offline_augmentations(train_orig, AUG_DIR)
    # 3) assemble train/val file lists
    train_files = list(train_orig)
    for orig in train_orig:
        label = orig.parent.name
        train_files.extend(sorted((AUG_DIR/label).glob(f"{orig.stem}_*.npy")))
    val_files = list(val_orig)
    # 4) DataLoaders
    train_dl = DataLoader(
        MelSpecDataset(train_files, SpecAugment()),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_dl = DataLoader(
        MelSpecDataset(val_files, None),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    # 5) model setup
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = LSTMAudioClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = step_epoch(model, train_dl, criterion, optimizer, device, EPSILON)
        val_loss,val_acc= step_epoch(model, val_dl, criterion, None,       device, 0.0)
        print(f"Epoch {epoch:02d} | train acc {tr_acc:.3f} | val acc {val_acc:.3f} | "
              f"train loss {tr_loss:.3f} | val loss {val_loss:.3f}")
        metrics = evaluate(model, val_dl, device)
        if not np.isnan(metrics['roc_auc']) and metrics['roc_auc']>best_auc:
            best_auc = metrics['roc_auc']
            ckpt     = CHECKPOINT_DIR/f"best_auc_{best_auc:.3f}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, ckpt)
            print(f"[CHECKPOINT] Saved {ckpt.name}")
    # 6) final eval & plots
    final = evaluate(model, val_dl, device)
    print("--- FINAL METRICS ---")
    for k,v in final.items():
        if not isinstance(v, np.ndarray): print(f"{k:10}: {v}")
    plot_confusion_matrix(final['cm'], STATS_DIR/'confusion_matrix.png')
    plot_roc_curve(final['y_true'], final['probs'], STATS_DIR/'roc_curve.png')
    print("Plots saved in artifacts/plots/")

if __name__ == '__main__':
    main()
