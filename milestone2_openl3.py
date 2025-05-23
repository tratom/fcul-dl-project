import argparse, os, json
from pathlib import Path
from datetime import datetime

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import openl3

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split

# ---------- FOLDER CONFIG ----------
DATA_ROOT = Path("data-source/audio")
CHECKPOINT_DIR = Path("artifacts/checkpoints")
PLOT_DIR = Path("artifacts/plots")
STATS_DIR = Path("artifacts/stats")
for d in (CHECKPOINT_DIR, PLOT_DIR, STATS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- SETTINGS ----------
MAX_SECONDS = 4.0
RANDOM_SEED = 42
EMBEDDING_SIZE = 2048

# ---------- LOAD OpenL3 MODEL ----------
def load_openl3_model():
    model = openl3.models.load_audio_embedding_model(
        input_repr="mel256",
        content_type="speech",
        embedding_size=512
    )
    return model

# ---------- FEATURE EXTRACTION ----------
def extract_openl3_embeddings(file_path, model):
    try:
        wav, sr = sf.read(file_path)
        max_len = int(MAX_SECONDS * sr)
        if len(wav) < max_len:
            wav = np.pad(wav, (0, max_len - len(wav)))
        else:
            wav = wav[:max_len]

        if wav.ndim == 1:
            wav = wav[:, np.newaxis]  # (T,) → (T, 1)

        emb, ts = openl3.get_audio_embedding(
            wav, sr, model,
            hop_size=0.1, center=True, verbose=0
        )

        if emb.shape[0] < 4:
            pad = np.zeros((4 - emb.shape[0], 512))
            emb = np.vstack([emb, pad])
        else:
            emb = emb[:4]

        return emb.flatten()  # (4, 512) → (2048,)
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        return None

# ---------- DATASET ----------
def load_dataset(files, model):
    X, y, speakers = [], [], []
    for path in files:
        emb = extract_openl3_embeddings(str(path), model)
        if emb is not None and emb.shape == (2048,):
            X.append(emb)
            y.append(0 if path.parent.name.startswith("HC") else 1)
            speakers.append(path.stem.split("_")[0])
    return np.array(X), np.array(y), np.array(speakers)

# ---------- METRICS ----------
def compute_and_save_metrics(y_true, y_prob, out_prefix):
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist()
    }

    # Save JSON
    with open(CHECKPOINT_DIR / f"{out_prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion Matrix
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["HC", "PD"])
    plt.yticks([0, 1], ["HC", "PD"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{out_prefix}_confusion_matrix.png")
    plt.close()

    # ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{out_prefix}_roc_curve.png")
        plt.close()
    except Exception:
        pass

    return metrics

# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--comments", type=str, default="")
    args = parser.parse_args()

    print(f"EXECUTION TIME: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("[INFO] Starting OpenL3 embedding classification.")
    if args.comments:
        print("[COMMENT]", args.comments)

    # List files
    all_files = []
    for sub in ("HC_AH", "PD_AH"):
        all_files.extend((DATA_ROOT / sub).glob("*.wav"))
    all_files = sorted(all_files, key=lambda p: p.name)
    labels = [0 if f.parent.name.startswith("HC") else 1 for f in all_files]

    train_f, val_f = train_test_split(
        all_files, test_size=0.3,
        stratify=labels, random_state=RANDOM_SEED
    )

    model = load_openl3_model()
    print("[INFO] Extracting features...")
    X_train, y_train, _ = load_dataset(train_f, model)
    X_val, y_val, _ = load_dataset(val_f, model)

    print("[INFO] Training classifier...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_val)[:, 1]

    metrics = compute_and_save_metrics(y_val, y_prob, out_prefix="openl3")

    print("--- FINAL VALIDATION METRICS ---")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k:10}: {v:.4f}")

    print("[INFO] Done.")

if __name__ == "__main__":
    main()
