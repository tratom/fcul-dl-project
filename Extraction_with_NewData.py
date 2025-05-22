import parselmouth
from parselmouth.praat import call
from pathlib import Path

import librosa
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pandas as pd

# === Percorsi ===
PD_PATH = Path('data-source/audio/PD_AH')
HC_PATH = Path('data-source/audio/HC_AH')
SPEC_PATH = Path('artifacts/mel_specs')

OUTPUT_PATH = Path('artifacts')

# === Funzione per estrarre feature da un file audio ===
def extract_features(file_path):
    snd = parselmouth.Sound(str(file_path))

    # pitch & harmonicity
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    pointProcess = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)

    # shimmer & jitter
    local_jitter = call(pointProcess, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    local_abs_jitter = call(pointProcess, "Get jitter (local, absolute)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    rap_jitter = call(pointProcess, "Get jitter (rap)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    ppq5_jitter = call(pointProcess, "Get jitter (ppq5)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    ddp_jitter = call(pointProcess, "Get jitter (ddp)", 0.0, 0.0, 0.0001, 0.02, 1.3)

    local_shimmer = call([snd, pointProcess], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    localdb_shimmer = call([snd, pointProcess], "Get shimmer (local_dB)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    apq3_shimmer = call([snd, pointProcess], "Get shimmer (apq3)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    apq5_shimmer = call([snd, pointProcess], "Get shimmer (apq5)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    apq11_shimmer = call([snd, pointProcess], "Get shimmer (apq11)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    dda_shimmer = call([snd, pointProcess], "Get shimmer (dda)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)

    # harmonic-to-noise ratio
    hnr = call(harmonicity, "Get mean", 0.0, 0.0)

    # pitch stats
    meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")
    stdevF0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")

    return {
        "meanF0": meanF0,
        "stdevF0": stdevF0,
        "hnr": hnr,
        "localJitter": local_jitter,
        "localabsoluteJitter": local_abs_jitter,
        "rapJitter": rap_jitter,
        "ppq5Jitter": ppq5_jitter,
        "ddpJitter": ddp_jitter,
        "localShimmer": local_shimmer,
        "localdbShimmer": localdb_shimmer,
        "apq3Shimmer": apq3_shimmer,
        "apq5Shimmer": apq5_shimmer,
        "apq11Shimmer": apq11_shimmer,
        "ddaShimmer": dda_shimmer,
    }

# === Analizza tutti i file PD.wav ===
folder =  PD_PATH
results = []

for wav_file in folder.glob("*.wav"):
    features = extract_features(wav_file)
    features["Label"] = "PD"
    results.append(features)

# === Salva per analisi o ML ===
df_pd = pd.DataFrame(results)

# === Analizza tutti i file HC.wav ===
folder =  HC_PATH
results = []

for wav_file in folder.glob("*.wav"):
    features = extract_features(wav_file)
    features["Label"] = "HC"
    results.append(features)

# === Salva per analisi o ML ===
df_hc = pd.DataFrame(results)
df_features = pd.concat([df_hc, df_pd], axis=0)

# 7. Salva solo il file finale
df_features.to_csv(OUTPUT_PATH / "vsp_all_features22.csv", index=False)