# Early Parkinson's Detection Using Speech Analysis
## Course project for the Deep Learning course at Facultade de Ciencias, Universidade de Lisboa.
### Group 8:
- Antonio Alampi (64316)
- Tommaso Tragno (64699)
- Cristian Tedesco (65149)
- Pol Rion Solé (65177)

The goal is to build a model that can detect early signs of Parkinson's disease using speech analysis.
The dataset consists of audio recordings of patients with Parkinson's disease and healthy controls.
The dataset is available at [FigShare](https://figshare.com/articles/dataset/Voice_Samples_for_Patients_with_Parkinson_s_Disease_and_Healthy_Controls/23849127)

The dataset is divided into two folders: HC_AH (Healthy Control) and PD_AH (Parkinson's Disease).
The audio files are in WAV format and have a sample rate of 8kHz.

## Folder layout
----------------------
```
project-root/
 ├─ data-source/
 │   └─ audio/
 │      ├─ HC_AH/   (Healthy Control - 41 wav)
 │      └─ PD_AH/   (Parkinson's Disease - 40 wav)
 ├─ artifacts/
 │   ├─ durations/
 │   │   ├─ audio_metadata.csv
 │   │   └─ duration_distribution.png
 │   ├─ mel_specs/
 │   │   └─ HC_AH_*.npy or PD_AH_*.npy   (fixed-length (MAX_FRAMES, N_MELS))
 │   ├─ plots/
 │   │   └─ HC_AH_*.png or PD_AH_*.png   (spectrogram plots for debugging)
 |   ├─ stats/      (Statistics and plots about training)
 │   └─ vsp_all_features_with_demographics.csv
 └─ milestone1_skeleton.py
```

# Reference Paper:

## 1. Predictive models in the diagnosis of Parkinson’s disease through voice analysis by Tomás Freitas Gonçalves (UPorto)
   ### Research Questions:
   
     1) Can automatically learned features — different from traditional ones such as Jitter, Shimmer, HNR, or MFCCs — extracted using deep neural networks (e.g., Wav2Vec, YAMNet,               Whisper, etc.) and used in machine learning models (e.g., SVM, Logistic Regression, Random Forest) lead to better predictive performance?
     2) Are deep learning models practically usable in clinical and medical settings?

   ### Strategy:

     1) Deep feature extraction + classical classifier

   ### Proposals for Implementation:
   
     1) Use both deep neural networks (DNNs) and pre-trained models (such as Wav2Vec, etc.) directly for the classification of Parkinson’s disease from raw audio data. 