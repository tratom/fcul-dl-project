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
        description="DataAugmentation Audio: generazione offline delle augmentazioni per training, validation e test"
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Cartella con sottocartelle training, validation, test (ognuna con HC_AH e PD_AH)"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data_augmented"),
        help="Cartella di destinazione per i WAV augmentati"
    )
    args = parser.parse_args()

    splits = ["training", "validation", "test"]
    for split in splits:
        # Raccogli file originali per questo split
        orig_paths: list[Path] = []
        split_dir = args.data_root / split
        for label in ["HC_AH", "PD_AH"]:
            orig_paths += sorted((split_dir / label).glob("*.wav"))

        # Directory di output per questo split
        split_out = args.out_dir / split
        print(f"Generazione augmentazioni per il split '{split}' ({len(orig_paths)} file)...")
        generate_offline_augmentations(orig_paths, split_out)
        print(f"Augmentazioni completate per '{split}'.\n")

if __name__ == "__main__":
    main()