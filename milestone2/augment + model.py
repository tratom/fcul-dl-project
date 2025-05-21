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
AUG_NPY_DIR    = Path("artifacts/augmented_audio_npy")
AUG_WAV_DIR    = Path("artifacts/augmented_audio_wav")
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
    b0 = b1 = b2 = 0.0
    out = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        white = np.random.randn()
        b0 = 0.99886 * b0 + white * 0.0555179
        b1 = 0.99332 * b1 + white * 0.0750759
        b2 = 0.96900 * b2 + white * 0.1538520
        out[i] = b0 + b1 + b2 + white * 0.5362
    out = out / np.sqrt(np.mean(out**2))
    return out

# -------------------- OFFLINE AUGMENTATIONS --------------------
def generate_offline_augmentations(orig_paths: List[Path], npy_dir: Path, wav_dir: Path) -> None:
    snr_db = 20
    speed_rates = [0.95, 1.05]
    pitch_steps = [-2, 2]
    for wav_path in orig_paths:
        label = wav_path.parent.name
        out_npy_dir = npy_dir / label
        out_wav_dir = wav_dir / label
        out_npy_dir.mkdir(parents=True, exist_ok=True)
        out_wav_dir.mkdir(parents=True, exist_ok=True)

        y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        librosa.output.write_wav(out_wav_dir / f"{wav_path.stem}_orig.wav", y, sr=SAMPLE_RATE)

        for rate, suffix in zip(speed_rates, ["speed_down", "speed_up"]):
            y_sp = librosa.effects.time_stretch(y, rate=rate)
            np.save(out_npy_dir / f"{wav_path.stem}_{suffix}.npy", y_sp)

        rms_signal = np.sqrt(np.mean(y**2))
        rms_noise = rms_signal / (10**(snr_db / 20))
        pink = generate_pink_noise(y.shape[0]) * rms_noise
        y_no = y + pink
        np.save(out_npy_dir / f"{wav_path.stem}_noise.npy", y_no)

        for step, suffix in zip(pitch_steps, ["pitch_down", "pitch_up"]):
            y_ps = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=step)
            np.save(out_npy_dir / f"{wav_path.stem}_{suffix}.npy", y_ps)

# (il resto dello script rimane invariato... solo main() viene aggiornato)

def main() -> None:
    parser = argparse.ArgumentParser(
        prog='milestone2.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Parkinson's Detection via Audio - Milestone 2"
    )
    parser.add_argument('-e','--epochs',type=int,default=10,help='Number of epochs')
    args = parser.parse_args()

    print(f"EXECUTION TIME: {datetime.now():%Y-%m-%d %H:%M:%S}")
    orig_paths: List[Path] = []
    for lbl in ('HC_AH','PD_AH'):
        orig_paths.extend(sorted((DATA_ROOT/lbl).glob('*.wav')))
    labels_orig = [0 if p.parent.name=='HC_AH' else 1 for p in orig_paths]
    train_orig,val_orig = train_test_split(
        orig_paths,test_size=0.3,stratify=labels_orig,random_state=RANDOM_SEED
    )

    AUG_NPY_DIR.mkdir(parents=True, exist_ok=True)
    AUG_WAV_DIR.mkdir(parents=True, exist_ok=True)
    generate_offline_augmentations(train_orig, AUG_NPY_DIR, AUG_WAV_DIR)

    train_files = list(train_orig)
    for orig in train_orig:
        label = orig.parent.name
        train_files.extend(sorted((AUG_NPY_DIR/label).glob(f"{orig.stem}_*.npy")))
    val_files = list(val_orig)

    train_dl = DataLoader(
        MelSpecDataset(train_files, SpecAugment()),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_dl = DataLoader(
        MelSpecDataset(val_files, None),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

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
    final = evaluate(model, val_dl, device)
    print("--- FINAL METRICS ---")
    for k,v in final.items():
        if not isinstance(v, np.ndarray): print(f"{k:10}: {v}")
    plot_confusion_matrix(final['cm'], STATS_DIR/'confusion_matrix.png')
    plot_roc_curve(final['y_true'], final['probs'], STATS_DIR/'roc_curve.png')
    print("Plots saved in artifacts/plots/")

if __name__ == '__main__':
    main()
