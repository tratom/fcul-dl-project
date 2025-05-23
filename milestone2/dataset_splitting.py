import os
import shutil
import random
from pathlib import Path

def prepare_datasets(
    raw_audio_dir: Path,
    aug_dirs: list[Path],
    prefix_filter: str,
    n_per_class: int,
    validation_dir: Path,
    training_dir: Path,
    seed: int = 42
):
    random.seed(seed)

    # individua le classi (sottocartelle es. HC_AH, PD_AH)
    classes = [d.name for d in raw_audio_dir.iterdir() if d.is_dir()]

    # 1) selezione stratificata per validation
    val_files = []
    for cls in classes:
        cls_dir = raw_audio_dir / cls
        candidates = [f for f in cls_dir.iterdir() 
                      if f.is_file() and f.name.startswith(prefix_filter)]
        if len(candidates) < n_per_class:
            raise ValueError(f"Non ci sono abbastanza file '{prefix_filter}*' in {cls_dir}")
        sampled = random.sample(candidates, n_per_class)
        val_files.extend(sampled)

    # crea cartelle di output
    for d in (validation_dir, training_dir):
        d.mkdir(exist_ok=True)

    # 2) copia validation (mantiene sottocartelle)
    for src in val_files:
        dst = validation_dir / src.parent.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / src.name)

    # 3) copia raw restanti in training
    for cls in classes:
        cls_dir = raw_audio_dir / cls
        train_cls_dir = training_dir / cls
        train_cls_dir.mkdir(parents=True, exist_ok=True)
        for f in cls_dir.iterdir():
            if f.is_file() and f not in val_files:
                shutil.copy2(f, train_cls_dir / f.name)

    # 4) copia augmented in training, escludendo quelli dei validation
    val_stems = {f.stem for f in val_files}
    for aug_dir in aug_dirs:
        # ad es. artifacts/augmented_audio_npy/HC_AH
        for cls in aug_dir.iterdir():
            if not cls.is_dir(): 
                continue
            for f in cls.iterdir():
                if not f.is_file():
                    continue
                # se nel nome compare uno degli stem di validation, skip
                if any(stem in f.name for stem in val_stems):
                    continue
                # altrimenti copia mantenendo struttura
                dst = training_dir / aug_dir.name / cls.name
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst / f.name)

if __name__ == "__main__":
    # configurazione dei percorsi
    raw_audio = Path("data-source/audio")
    aug_npy   = Path("artifacts/augmented_audio_npy")
    aug_wav   = Path("artifacts/augmented_audio_wav")
    val_dir   = Path("validation")
    train_dir = Path("training")

    prepare_datasets(
        raw_audio_dir=raw_audio,
        aug_dirs=[aug_npy, aug_wav],
        prefix_filter="AH",
        n_per_class=16,
        validation_dir=val_dir,
        training_dir=train_dir,
        seed=123
    )
    print("Dataset preparati in 'validation/' e 'training/'")
