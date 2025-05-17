# ------------------------------------------------------------------------------
# Script di preprocessing audio → immagini per sanity‐check Tiny CNN
#
# 1) Carica i file WAV di HC_AH e PD_AH
# 2) Normalizza l’onda e trimma le code di silenzio
#    usando un criterio di “discesa” del segnale
# 3) Estrae i log-mel spectrogram (64 bande, hop 10 ms, win 25 ms)
# 4) Pad/Trunca ogni spectrogram a 1024 frame (≈10 s) per uniformità
# 5) Salva i risultati come PNG in artifacts/plots/HC_AH/ e artifacts/plots/PD_AH/
#    pronti per il training della Tiny CNN (Conv2d → AdaptiveAvgPool2d → Linear)
# ------------------------------------------------------------------------------
#Barra a sinistra: è il segmento “voiced” dopo il trim by-descent, 
# con tutte le frequenze visibili nel tempo.
#Spazio nero a destra: è il padding residuo (–80 dB) fino a 1024 frame, 
# che serve a mantenere tutte le immagini della stessa dimensione.
#
#
import os
from pathlib import Path

import numpy as np
import librosa
import matplotlib.pyplot as plt

# ---------- Trim function based on sample-to-sample descent ----------
def trim_segment_by_descent(segment: np.ndarray,
                             sr: int,
                             threshold: float = 0.1,
                             end_buffer: float = 0.075) -> np.ndarray:
    """
    Trims the end of an audio segment by finding the first point where the
    successive-sample difference becomes negative beyond a fraction of the segment length,
    then truncates the segment including a small time buffer.
    """
    diff = np.diff(segment)
    n = len(segment)
    start_idx = int(threshold * n)
    neg_indices = np.where(diff < 0)[0]
    descent_points = [i for i in neg_indices if i >= start_idx]
    if not descent_points:
        return segment
    buffer_samples = int(end_buffer * sr)
    cut_idx = descent_points[0] + buffer_samples
    cut_idx = min(cut_idx, n)
    return segment[:cut_idx]

# ---------- CONFIGURATION ----------
SAMPLE_RATE = 16_000        # Hz
N_MELS      = 64
HOP_LENGTH  = 160           # samples (10 ms)
WIN_LENGTH  = 400           # samples (25 ms)
FMIN        = 50
FMAX        = 4_000
MAX_FRAMES  = 1_024         # ≈10 s @ 100 fps

# Input audio directories
audio_dirs = {
    "HC_AH": Path("data-source/audio/HC_AH"),
    "PD_AH": Path("data-source/audio/PD_AH"),
}

# Output directory for spectrogram images
OUTPUT_DIR = Path("milestone2/plots_padding")
for label in audio_dirs:
    (OUTPUT_DIR / label).mkdir(parents=True, exist_ok=True)

# ---------- PROCESSING LOOP ----------
for label, in_dir in audio_dirs.items():
    for wav_path in in_dir.glob("*.wav"):
        # 1) load & normalize
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        y = librosa.util.normalize(y)
        # 2) trim silences via descent-based
        y_trim = trim_segment_by_descent(y, sr,
                                         threshold=0.1,
                                         end_buffer=0.075)
        # 3) compute mel-spectrogram
        melspec = librosa.feature.melspectrogram(
            y=y_trim,
            sr=sr,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            fmin=FMIN,
            fmax=FMAX,
            power=2.0,
        )
        # 4) to log scale (dB) and transpose → (T, M)
        logmel = librosa.power_to_db(melspec, ref=np.max).T.astype(np.float32)

        # 5) pad or truncate to MAX_FRAMES
        T = logmel.shape[0]
        if T >= MAX_FRAMES:
            logmel = logmel[:MAX_FRAMES]
        else:
            pad = MAX_FRAMES - T
            logmel = np.pad(logmel,
                            ((0, pad), (0, 0)),
                            mode="constant",
                            constant_values=-80.0)

        # 6) plot & save as PNG
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        # display transposed so freq on y-axis, time on x-axis
        ax.imshow(logmel.T,
                  aspect="auto",
                  origin="lower",
                  cmap="magma")
        ax.axis("off")
        out_name = wav_path.stem + ".png"
        out_path = OUTPUT_DIR / label / out_name
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

print(f"All spectrogram images saved under: {OUTPUT_DIR.resolve()}")  
