import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# Funzione di trimming basata sulla discesa del segnale

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

# Configurazione input/output
audio_dirs = {
    'HC_AH': os.path.join('data-source', 'audio', 'HC_AH'),
    'PD_AH': os.path.join('data-source', 'audio', 'PD_AH'),
}
output_base = 'trimmed_audio'

# Crea cartelle di output per ciascuna label
for label in audio_dirs:
    out_dir = os.path.join(output_base, f"{label}_trimmed")
    os.makedirs(out_dir, exist_ok=True)

# Parametri trim
threshold = 0.1       # 10% della durata
end_buffer = 0.075    # 75 ms di buffer

# Processo batch
tqdm_iter = tqdm(audio_dirs.items(), desc="Labels")
for label, in_dir in tqdm_iter:
    files = [f for f in os.listdir(in_dir) if f.lower().endswith('.wav')]
    tqdm_files = tqdm(files, desc=f"Processing {label}")
    for fname in tqdm_files:
        in_path = os.path.join(in_dir, fname)
        try:
            y, sr = librosa.load(in_path, sr=None)
            trimmed = trim_segment_by_descent(y, sr, threshold, end_buffer)
            out_dir = os.path.join(output_base, f"{label}_trimmed")
            out_name = f"{label}_{fname}"
            out_path = os.path.join(out_dir, out_name)
            sf.write(out_path, trimmed, sr)
        except Exception as e:
            print(f"Errore {fname} ({label}): {e}")

print(f"All done! Trimmed files saved under '{output_base}/' with label-specific folders.")