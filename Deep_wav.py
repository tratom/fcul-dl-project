import librosa
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pandas as pd

# === Percorsi ===
PD_PATH = Path('C:/Users/cri_t/OneDrive/Desktop/Lisboa_esami/data_deep/Data/Data1/PD_AH')
HC_PATH = Path('C:/Users/cri_t/OneDrive/Desktop/Lisboa_esami/data_deep/Data/Data1/HC_AH')

OUTPUT_PD = Path('C:/Users/cri_t/OneDrive/Desktop/Lisboa_esami/data_deep/Data/Specto_pd')
OUTPUT_HC = Path('C:/Users/cri_t/OneDrive/Desktop/Lisboa_esami/data_deep/Data/Specto_hc')

#OUTPUT_PD.mkdir(parents=True, exist_ok=True)
#OUTPUT_HC.mkdir(parents=True, exist_ok=True)

# === Funzione di preprocessing ===
#def process_audio_files(input_path, output_path):
    #raw_audio = os.listdir(input_path)
    #for wav in tqdm(raw_audio, desc=f"Processing {input_path.name}"):
        #file_path = input_path / wav
        #if not file_path.suffix == '.wav':
            #continue  # Salta file non audio
        #y, sr = librosa.load(file_path, sr=16000)
        #y = librosa.util.normalize(y)
        #mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        #np.save(output_path / (Path(wav).stem + ".npy"), mel)

# === Esegui preprocessing con barra di avanzamento ===
#process_audio_files(PD_PATH, OUTPUT_PD)
#process_audio_files(HC_PATH, OUTPUT_HC)

# === Visualizza un esempio sano ===
pippo_sano = np.load(OUTPUT_HC / 'AH_121A_BD5BA248-E807-4CB9-8B53-47E7FFE5F8E2.npy')
print("Dimensioni paziente sano:", pippo_sano.shape)
plt.figure(figsize=(10, 4))
plt.imshow(pippo_sano, aspect='auto', origin='lower')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram Sano')
plt.xlabel('Time')
plt.ylabel('Mel frequency bins')
plt.tight_layout()
plt.show()

# === Visualizza un esempio malato ===
pippo = np.load(OUTPUT_PD / 'AH_545629296-C2C009C6-8C17-42EA-B6BE-362942FC4692.npy')
print("Dimensioni paziente malato:", pippo.shape)
plt.figure(figsize=(10, 4))
plt.imshow(pippo, aspect='auto', origin='lower')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram Malato')
plt.xlabel('Time')
plt.ylabel('Mel frequency bins')
plt.tight_layout()
plt.show()


import parselmouth
from parselmouth.praat import call
from pathlib import Path

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
#folder =  PD_PATH
#results = []

#for wav_file in folder.glob("*.wav"):
    #features = extract_features(wav_file)
    #features["filename"] = wav_file.name
    #results.append(features)

# === Salva in CSV per analisi o ML ===
#df = pd.DataFrame(results)
#df.to_csv("vsp_pd_features.csv", index=False)

# === Analizza tutti i file HC.wav ===
#folder =  HC_PATH
#results = []

#for wav_file in folder.glob("*.wav"):
    #features = extract_features(wav_file)
    #features["filename"] = wav_file.name
    #results.append(features)

# === Salva in CSV per analisi o ML ===
#df = pd.DataFrame(results)
#df.to_csv("vsp_hc_features.csv", index=False)




#df_features = pd.read_csv("C:/Users/cri_t/OneDrive/Desktop/Lisboa_esami/data_deep/vsp_hc_features.csv")
#df_demo = pd.read_excel("C:/Users/cri_t/OneDrive/Desktop/Lisboa_esami/data_deep/Demographics_age_sex.xlsx")


#df_features["Sample ID"] = df_features["filename"].str.replace(".wav", "", regex=False)
#df_merged = pd.merge(df_features, df_demo, on="Sample ID", how="left")
#df_merged.to_csv("vsp_hc_features_with_demographics.csv", index=False)



#df_features = pd.read_csv("C:/Users/cri_t/OneDrive/Desktop/Lisboa_esami/data_deep/vsp_pd_features.csv")
#df_demo = pd.read_excel("C:/Users/cri_t/OneDrive/Desktop/Lisboa_esami/data_deep/Demographics_age_sex.xlsx")


#df_features["Sample ID"] = df_features["filename"].str.replace(".wav", "", regex=False)
#df_merged = pd.merge(df_features, df_demo, on="Sample ID", how="left")
#df_merged.to_csv("vsp_pd_features_with_demographics.csv", index=False)




# Carica entrambi i file
#df_pd = pd.read_csv("C:/Users/cri_t/OneDrive/Desktop/Lisboa_esami/data_deep/vsp_pd_features_with_demographics.csv")
#df_hc = pd.read_csv("C:/Users/cri_t/OneDrive/Desktop/Lisboa_esami/data_deep/vsp_hc_features_with_demographics.csv")

# Unisci i due DataFrame
#df_combined = pd.concat([df_pd, df_hc], ignore_index=True)

# Salva il risultato
#df_combined.to_csv("vsp_all_features_with_demographics.csv", index=False)










