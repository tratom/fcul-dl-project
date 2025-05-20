#!/usr/bin/env python3
"""
Script di Augmentazione Audio - Solo Data Augmentation
Genera offline 5 versioni WAV per ogni file originale nel training set:
  - speed_down (-5%)
  - speed_up   (+5%)
  - pink noise (20 dB SNR)
  - pitch_down (-2 semitoni)
  - pitch_up   (+2 semitoni)
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import List
import os
import librosa
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split

# Configurazione di default
SAMPLE_RATE = 16_000   # 16 kHz
SNR_DB      = 20       # dB per pink noise
SPEED_RATES = [0.95, 1.05]
PITCH_STEPS = [-2, 2]


def generate_pink_noise(n_samples: int) -> np.ndarray:
    """
    Genera rumore rosa 1/f usando l'algoritmo di Paul Kellet.
    Normalizza RMS a 1.
    """
    b0 = b1 = b2 = 0.0
    out = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        white = np.random.randn()
        b0 = 0.99886 * b0 + white * 0.0555179
        b1 = 0.99332 * b1 + white * 0.0750759
        b2 = 0.96900 * b2 + white * 0.1538520
        out[i] = b0 + b1 + b2 + white * 0.5362
    return out / np.sqrt(np.mean(out**2))


def generate_offline_augmentations(orig_paths: List[Path], output_dir: Path) -> None:
    """
    Per ogni WAV in orig_paths genera e salva 5 augmentazioni
    come file WAV in output_dir/<label>/
    """
    for wav_path in orig_paths:
        label_dir = output_dir / wav_path.parent.name
        label_dir.mkdir(parents=True, exist_ok=True)
        y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        # 1) speed perturb
        for rate, suffix in zip(SPEED_RATES, ["speed_down", "speed_up"]):
            y_sp = librosa.effects.time_stretch(y, rate=rate)
            sf.write(label_dir / f"{wav_path.stem}_{suffix}.wav", y_sp, SAMPLE_RATE)
        # 2) pink noise
        rms = np.sqrt(np.mean(y**2))
        noise = generate_pink_noise(len(y)) * (rms / (10**(SNR_DB/20)))
        y_no = y + noise
        sf.write(label_dir / f"{wav_path.stem}_noise.wav", y_no, SAMPLE_RATE)
        # 3) pitch shift
        for step, suffix in zip(PITCH_STEPS, ["pitch_down", "pitch_up"]):
            y_ps = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=step)
            sf.write(label_dir / f"{wav_path.stem}_{suffix}.wav", y_ps, SAMPLE_RATE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solo DataAugmentation Audio: generazione offline delle augmentazioni per il training set"
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data-source/audio"),
        help="Cartella con sottocartelle HC_AH e PD_AH contenenti WAV originali"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("artifacts/augmented_audio"),
        help="Cartella di destinazione per i WAV augmentati"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.3,
        help="Frazione di file originali da tenere per validation (no augmentazioni)"
    )
    args = parser.parse_args()

    # 1) Lista dei file originali
    orig_paths: List[Path] = []
    for label in ["HC_AH", "PD_AH"]:
        orig_paths += sorted((args.data_root / label).glob("*.wav"))

    # 2) Split train/validation sui file originali
    labels = [0 if p.parent.name == "HC_AH" else 1 for p in orig_paths]
    train_orig, val_orig = train_test_split(
        orig_paths, test_size=args.test_size,
        stratify=labels, random_state=RANDOM_SEED
    )

    # 3) Genera augmentazioni SOLO per il training set
    print(f"Generazione augmentazioni per {len(train_orig)} file di training...")
    generate_offline_augmentations(train_orig, args.out_dir)
    print("Augmentazioni completate.")
    print(f"Training originals: {len(train_orig)}, Validation originals (no augment): {len(val_orig)}")

if __name__ == "__main__":
    main()
