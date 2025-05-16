import os
import pandas as pd
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

# === Configurazione percorsi ===
# Mappa etichette semplici ('HC', 'PD') alle cartelle dei file trimmati
trimmed_dirs = {
    'HC': os.path.join('trimmed_audio', 'HC_AH_trimmed'),
    'PD': os.path.join('trimmed_audio', 'PD_AH_trimmed'),
}
output_csv = os.path.join('trimmed_audio', 'trimmed_audio_features.csv')

# === Controllo esistenza cartelle ===
for label, d in trimmed_dirs.items():
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Directory non trovata: {d}")

# === Funzioni per estrazione feature ===
def extract_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)

def extract_parselmouth_features(wav_path: str) -> dict:
    snd = parselmouth.Sound(wav_path)
    # Pitch & harmonicity
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    # Jitter
    local_jitter = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    local_abs_jitter = call(point_process, "Get jitter (local, absolute)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    rap_jitter = call(point_process, "Get jitter (rap)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    ppq5_jitter = call(point_process, "Get jitter (ppq5)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    ddp_jitter = call(point_process, "Get jitter (ddp)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    # Shimmer
    local_shimmer = call([snd, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    localdb_shimmer = call([snd, point_process], "Get shimmer (local_dB)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    apq3_shimmer = call([snd, point_process], "Get shimmer (apq3)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    apq5_shimmer = call([snd, point_process], "Get shimmer (apq5)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    apq11_shimmer = call([snd, point_process], "Get shimmer (apq11)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    dda_shimmer = call([snd, point_process], "Get shimmer (dda)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    # HNR & pitch stats
    hnr = call(harmonicity, "Get mean", 0.0, 0.0)
    meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")
    stdevF0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")

    return {
        "meanF0": meanF0,
        "stdevF0": stdevF0,
        "hnr": hnr,
        "localJitter": local_jitter,
        "localAbsJitter": local_abs_jitter,
        "rapJitter": rap_jitter,
        "ppq5Jitter": ppq5_jitter,
        "ddpJitter": ddp_jitter,
        "localShimmer": local_shimmer,
        "localDbShimmer": localdb_shimmer,
        "apq3Shimmer": apq3_shimmer,
        "apq5Shimmer": apq5_shimmer,
        "apq11Shimmer": apq11_shimmer,
        "ddaShimmer": dda_shimmer,
    }

# === Elaborazione batch ===
records = []
for label, in_dir in tqdm(trimmed_dirs.items(), desc="Labels"):
    for wav_file in tqdm(os.listdir(in_dir), desc=f"Files {label}", leave=False):
        if not wav_file.lower().endswith('.wav'):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        # MFCC
        y, sr = librosa.load(wav_path, sr=None)
        mfcc_vals = extract_mfcc(y, sr)
        # Parselmouth features
        pm_feats = extract_parselmouth_features(wav_path)
        # Costruisci record
        rec = {
            'wav_file': wav_file,
            'label': label,  # 'HC' o 'PD'
            'wav_path': wav_path,
        }
        # Aggiungi MFCC
        for i, v in enumerate(mfcc_vals, start=1):
            rec[f'mfcc{i}'] = v
        # Aggiungi parselmouth
        rec.update(pm_feats)
        records.append(rec)

# Crea DataFrame e salva
df_feats = pd.DataFrame(records)
df_feats.to_csv(output_csv, index=False)
print(f"Features salvate in {output_csv}")
