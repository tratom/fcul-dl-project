#!/usr/bin/env python3
"""
Early Parkinson's Detection Using Speech Analysis - Waveform Augmentation Script
Genera:
 1) 5 versioni augmentate per ogni file audio:
    - speed down / speed up
    - pitch down / pitch up
    - pink noise
 2) Salva ciascuna versione in formato:
    - .wav in artifacts/augmented_audio_wav
    - .npy in artifacts/augmented_audio_npy
"""
from __future__ import annotations
import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List

import librosa
import numpy as np
import soundfile as sf

# -------------------- CONFIG --------------------
DATA_ROOT     = Path("data-source/audio")
AUG_WAV_DIR   = Path("artifacts/augmented_audio_wav")
AUG_NPY_DIR   = Path("artifacts/augmented_audio_npy")
SAMPLE_RATE   = 16_000  # 16 kHz
RANDOM_SEED   = 42

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

# -------------------- AUGMENTATION FUNCTION --------------------
def generate_offline_augmentations(orig_paths: List[Path], wav_dir: Path, npy_dir: Path) -> None:
    snr_db = 20
    speed_rates = [0.95, 1.05]
    pitch_steps = [-2, 2]

    for wav_path in orig_paths:
        label = wav_path.parent.name
        out_wav_dir = wav_dir / label
        out_npy_dir = npy_dir / label
        out_wav_dir.mkdir(parents=True, exist_ok=True)
        out_npy_dir.mkdir(parents=True, exist_ok=True)

        y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)

        # 1) speed perturb
        for rate, suffix in zip(speed_rates, ["speed_down", "speed_up"]):
            y_sp = librosa.effects.time_stretch(y, rate=rate)
            sf.write(out_wav_dir / f"{wav_path.stem}_{suffix}.wav", y_sp, SAMPLE_RATE)
            np.save(out_npy_dir / f"{wav_path.stem}_{suffix}.npy", y_sp)

        # 2) pink noise @ 20 dB SNR
        rms_signal = np.sqrt(np.mean(y**2))
        rms_noise  = rms_signal / (10**(snr_db / 20))
        pink = generate_pink_noise(y.shape[0]) * rms_noise
        y_no = y + pink
        sf.write(out_wav_dir / f"{wav_path.stem}_noise.wav", y_no, SAMPLE_RATE)
        np.save(out_npy_dir / f"{wav_path.stem}_noise.npy", y_no)

        # 3) pitch shift
        for step, suffix in zip(pitch_steps, ["pitch_down", "pitch_up"]):
            y_ps = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=step)
            sf.write(out_wav_dir / f"{wav_path.stem}_{suffix}.wav", y_ps, SAMPLE_RATE)
            np.save(out_npy_dir / f"{wav_path.stem}_{suffix}.npy", y_ps)

# -------------------- MAIN --------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog='augment_waveforms.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Genera file WAV e NPY con 5 augmentazioni per ogni audio originale"
    )
    args = parser.parse_args()

    print(f"EXECUTION TIME: {datetime.now():%Y-%m-%d %H:%M:%S}")

    orig_paths: List[Path] = []
    for lbl in ('HC_AH', 'PD_AH'):
        orig_paths.extend(sorted((DATA_ROOT / lbl).glob('*.wav')))

    random.seed(RANDOM_SEED)
    AUG_WAV_DIR.mkdir(parents=True, exist_ok=True)
    AUG_NPY_DIR.mkdir(parents=True, exist_ok=True)

    generate_offline_augmentations(orig_paths, AUG_WAV_DIR, AUG_NPY_DIR)
    print(f"Augmentazioni completate. WAV in: {AUG_WAV_DIR}, NPY in: {AUG_NPY_DIR}")

if __name__ == '__main__':
    main()
