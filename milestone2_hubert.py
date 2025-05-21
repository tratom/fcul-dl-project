#!/usr/bin/env python3
"""
Milestone 3 – Early Parkinson's Detection Using Speech  (HuBERT base)
--------------------------------------------------------------------
Drop-in replacement for milestone2_wav2vec2.py that swaps in a
facebook/hubert-base-ls960 encoder and adds two ready-made experiments:

    • baseline  – freeze everything, train only the linear classifier head
    • last2     – freeze all, then un-freeze the last 2 encoder layers
                  (LR 1e-5, 10 epochs)

Usage
-----
python train_hubert_pd.py --exp baseline --epochs 3            # fast sanity-check  
python train_hubert_pd.py --exp last2                          # 10 epochs, LR 1e-5  
"""

from __future__ import annotations
import argparse, os, json, random, inspect
from pathlib import Path
from datetime import datetime

import numpy as np
import torch, torchaudio, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve,
)
from sklearn.model_selection import train_test_split

# ───────────────────────── HF imports (HuBERT) ────────────────────────────
from transformers import AutoProcessor, Trainer, TrainerCallback, HubertForSequenceClassification
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import set_seed
import transformers

# ---------- FOLDER CONFIG -------------------------------------------------
DATA_ROOT      = Path("data-source/audio")
CHECKPOINT_DIR = Path("artifacts/checkpoints")
PLOT_DIR       = Path("artifacts/plots")
STATS_DIR      = Path("artifacts/stats")
for d in (CHECKPOINT_DIR, PLOT_DIR, STATS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- AUDIO + MODEL CONFIG -----------------------------------------
SAMPLE_RATE = 16_000
MAX_SECONDS = 4.0                          # pad / truncate to 64 k frames
MODEL_ID    = "facebook/hubert-base-ls960" # ← HuBERT-base
RANDOM_SEED = 42
set_seed(RANDOM_SEED)

# ---------- DATASET -------------------------------------------------------
class PDVoiceDataset(Dataset):
    """Pad/truncate all utterances to a fixed length and return tensors
       ready for HuBERT (or Wav2Vec-style) models."""

    def __init__(self, files: list[Path], processor: HubertProcessor, augment=False):
        self.files, self.processor, self.augment = files, processor, augment
        self.labels  = [0 if f.parent.name.startswith("HC") else 1 for f in files]
        self.max_len = int(SAMPLE_RATE * MAX_SECONDS)

    def _load_audio(self, wav_path: Path):
        wav, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(dim=0)

        # -- simple augmentation (gain / white-noise) ----------------------
        if self.augment:
            if random.random() < 0.3:
                wav *= random.uniform(0.9, 1.1)
            if random.random() < 0.3:
                wav += 0.005 * torch.randn_like(wav)

        # -- pad or truncate ----------------------------------------------
        if len(wav) < self.max_len:
            wav = torch.nn.functional.pad(wav, (0, self.max_len - len(wav)))
        else:
            wav = wav[: self.max_len]
        return wav

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        wav = self._load_audio(self.files[idx])
        mask = torch.zeros(self.max_len, dtype=torch.long)
        mask[:len(wav)] = 1

        proc = self.processor(
            wav.numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_len,
        )
        return {
            "input_values"  : proc["input_values"].squeeze(0),
            "attention_mask": mask,
            "labels"        : torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ---------- METRICS -------------------------------------------------------
def compute_metrics(pred):
    logits = torch.tensor(pred.predictions)
    y_prob = torch.softmax(logits, dim=-1)[:, 1].numpy()
    y_pred = (y_prob > 0.5).astype(int)
    y_true = pred.label_ids

    acc  = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc_auc}

# ---------- PLOTS ---------------------------------------------------------
def plot_confusion_matrix(cm, out):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix"); plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["HC", "PD"]); plt.yticks(ticks, ["HC", "PD"])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(int(cm[i, j])),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout(); plt.savefig(out); plt.close()

def plot_roc(y_true, y_prob, out):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
    except ValueError: return
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2); plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve (AUC={auc:.3f})"); plt.tight_layout(); plt.savefig(out); plt.close()

# ---------- CALLBACK: un-freeze N top layers ------------------------------
class UnfreezeTopLayersCallback(TrainerCallback):
    """At epoch 0 re-enable the last *n_layers* transformer blocks."""
    def __init__(self, n_layers: int, lr: float):
        self.n_layers, self.lr, self.done = n_layers, lr, False

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.done or state.epoch > 0:                      # only once
            return
        model, optim = kwargs["model"], kwargs["optimizer"]
        base = getattr(model, "hubert", None)                 # HuBERT encoder
        new_params = []
        for layer in base.encoder.layers[-self.n_layers:]:
            for p in layer.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    new_params.append(p)
        optim.add_param_group({"params": new_params, "lr": self.lr, "weight_decay": 0.01})
        print(f"[Callback] Un-froze {self.n_layers} top layers, added LR {self.lr:.1e}")
        self.done = True

# ---------- UTILITIES -----------------------------------------------------
def list_audio_files():
    files = []
    for sub in ("HC_AH", "PD_AH"):
        files.extend((DATA_ROOT / sub).glob("*.wav"))
    return sorted(files, key=lambda p: p.name)

def freeze_all_but_head(model):
    for p in model.hubert.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if any(k in name for k in ("projector", "classifier")):
            p.requires_grad = True

# ---------- MAIN ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--exp", choices=["baseline", "last2"], required=True,
                    help="baseline: linear head only | last2: un-freeze last 2 encoder layers")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--plots", action="store_true", help="Save CM & ROC to artifacts/plots/")
    args = ap.parse_args()

    print(f"[TIME] {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"[INFO] Experiment: {args.exp}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device   :", device)
    print("[INFO] HF model :", MODEL_ID)
    print("Transformers ver:", transformers.__version__)

    # -- data --------------------------------------------------------------
    all_files = list_audio_files()
    labels    = [0 if f.parent.name.startswith("HC") else 1 for f in all_files]
    tr_f, val_f = train_test_split(all_files, test_size=0.3,
                                   stratify=labels, random_state=RANDOM_SEED)

    processor = AutoProcessor.from_pretrained(MODEL_ID, do_normalize=False)
    tr_ds = PDVoiceDataset(tr_f, processor, augment=True)
    val_ds = PDVoiceDataset(val_f, processor, augment=False)

    # -- model -------------------------------------------------------------
    model = HubertForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2,
        attention_dropout=0.05, hidden_dropout=0.05,
        gradient_checkpointing=torch.cuda.is_available(),
    )

    freeze_all_but_head(model)

    # EXP -- baseline vs last2 --------------------------------------------
    callbacks = []
    lr = 5e-4 if args.exp == "baseline" else 1e-5
    if args.exp == "last2":
        callbacks.append(UnfreezeTopLayersCallback(n_layers=2, lr=lr))
        args.epochs = 10                                                # enforce spec

    # -- training args ----------------------------------------------------
    train_args = TrainingArguments(
        output_dir                  = str(CHECKPOINT_DIR),
        per_device_train_batch_size = 4,
        per_device_eval_batch_size  = 4,
        learning_rate               = lr,
        num_train_epochs            = args.epochs,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "roc_auc",
        greater_is_better           = True,
        fp16                        = torch.cuda.is_available(),
        logging_steps               = 10,
        report_to                   = "none",
        max_grad_norm               = 1.0,
    )

    trainer = Trainer(
        model           = model,
        args            = train_args,
        train_dataset   = tr_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
        callbacks       = callbacks,
    )

    trainer.train()

    # -- evaluation & bookkeeping ----------------------------------------
    preds   = trainer.predict(val_ds)
    metrics = compute_metrics(preds)
    y_prob  = torch.softmax(torch.tensor(preds.predictions), dim=-1)[:, 1].numpy()
    y_true  = preds.label_ids
    cm      = confusion_matrix(y_true, (y_prob > 0.5).astype(int))
    metrics["confusion_matrix"] = cm.tolist()

    best_ckpt   = Path(trainer.state.best_model_checkpoint or CHECKPOINT_DIR)
    with open(best_ckpt / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[CHECKPOINT] Best model @ {best_ckpt}")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k:9}: {v}")

    # plots
    if args.plots:
        plot_confusion_matrix(cm, PLOT_DIR / "confusion_matrix.png")
        plot_roc(y_true, y_prob, PLOT_DIR / "roc_curve.png")
        print("[INFO] Plots saved to artifacts/plots/")

    print("[INFO] Training complete.")

if __name__ == "__main__":
    main()
