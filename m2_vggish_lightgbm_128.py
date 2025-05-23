import argparse, os, json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

from torchvggish import vggish, vggish_input

# ---------- FOLDER CONFIG ----------
DATA_ROOT = Path("data-source/audio")
CHECKPOINT_DIR = Path("artifacts/checkpoints")
PLOT_DIR = Path("artifacts/plots")
STATS_DIR = Path("artifacts/stats")
for d in (CHECKPOINT_DIR, PLOT_DIR, STATS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- SETTINGS ----------
SAMPLE_RATE = 16_000
MAX_SECONDS = 4.0
RANDOM_SEED = 42
EMBEDDING_SIZE = 128

# ---------- LOAD VGGish MODEL ----------
def load_vggish_model():
    model = vggish()
    model.eval()
    return model

# ---------- FEATURE EXTRACTION ----------
def extract_vggish_embeddings(file_path, model):
    try:
        wav, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        wav = wav[:int(MAX_SECONDS * SAMPLE_RATE)]
        if len(wav) < SAMPLE_RATE:
            wav = np.pad(wav, (0, SAMPLE_RATE - len(wav)))
        example = vggish_input.waveform_to_examples(wav, SAMPLE_RATE)
        with torch.no_grad():
            embedding = model(example if isinstance(example, torch.Tensor) else torch.from_numpy(example))
        if len(embedding.shape) > 1:
            return embedding.mean(dim=0).numpy()
        else:
            return embedding.numpy()
    except Exception as e:
        print(f"Error with {file_path}: {e}")
        return np.zeros((EMBEDDING_SIZE,), dtype=np.float32)

# ---------- DATASET ----------
def load_dataset(files, model):
    X, y, speakers = [], [], []
    for path in files:
        emb = extract_vggish_embeddings(str(path), model)
        if emb is not None:
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

    with open(CHECKPOINT_DIR / f"{out_prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

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
    print("[INFO] Starting VGGish + LightGBM classification.")
    if args.comments:
        print("[COMMENT]", args.comments)

    all_files = []
    for sub in ("HC_AH", "PD_AH"):
        all_files.extend((DATA_ROOT / sub).glob("*.wav"))
    all_files = sorted(all_files, key=lambda p: p.name)
    labels = [0 if f.parent.name.startswith("HC") else 1 for f in all_files]

    train_f, val_f = train_test_split(
        all_files, test_size=0.3,
        stratify=labels, random_state=RANDOM_SEED
    )

    model = load_vggish_model()
    print("[INFO] Extracting features...")
    X_train, y_train, _ = load_dataset(train_f, model)
    X_val, y_val, _ = load_dataset(val_f, model)

    print("X_train shape:", X_train.shape)
    print("X_train std per feature:", np.std(X_train, axis=0))
    print("Any constant feature?:", np.any(np.std(X_train, axis=0) == 0))

    print("[INFO] Training LightGBM classifier...")
    clf = LGBMClassifier(n_estimators=450, class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_val)[:, 1]

    metrics = compute_and_save_metrics(y_val, y_prob, out_prefix="vggish_lgbm")

    print("--- FINAL VALIDATION METRICS ---")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k:10}: {v:.4f}")

    print("[INFO] Done.")

if __name__ == "__main__":
    main()
