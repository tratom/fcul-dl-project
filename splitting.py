import os
import shutil
import random

# Percorsi di origine e destinazione
source_dir = 'data-source/audio'
classes = ['HC_AH', 'PD_AH']
target_root = 'data'  # Qui si creeranno le cartelle training, validation, test

# Percentuali per i vari split
splits = {
    'training': 0.8,
    'validation': 0.1,
    'test': 0.1
}

# Creazione delle cartelle di destinazione
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(target_root, split, cls), exist_ok=True)

# Divisione dei file per classe
for cls in classes:
    cls_source = os.path.join(source_dir, cls)
    wav_files = [f for f in os.listdir(cls_source) if f.lower().endswith('.wav')]
    random.shuffle(wav_files)
    total = len(wav_files)
    n_train = int(total * splits['training'])
    n_val = int(total * splits['validation'])
    # Calcolo del resto per il test
    n_test = total - n_train - n_val

    split_map = {
        'training': wav_files[:n_train],
        'validation': wav_files[n_train:n_train + n_val],
        'test': wav_files[n_train + n_val:]
    }

    for split, files in split_map.items():
        for filename in files:
            src = os.path.join(cls_source, filename)
            dst = os.path.join(target_root, split, cls, filename)
            shutil.copy2(src, dst)

    print(f"Classe {cls}: {n_train} training, {n_val} validation, {n_test} test files.")

