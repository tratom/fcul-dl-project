#!/usr/bin/env python3
"""
Milestone 2 - Early Parkinson's Detection Using Speech (Wav2Vec 2.0)
-------------------------------------------------------------------
Transfer-learning script that replaces the log-mel + LSTM baseline with a
pre-trained Wav2Vec 2.0 encoder and a tiny classification head.

✓ Keeps your folder structure:
    project-root/
      ├─ data-source/audio/HC_AH/*.wav
      ├─ data-source/audio/PD_AH/*.wav
      ├─ artifacts/checkpoints/
      ├─ artifacts/plots/
      └─ artifacts/stats/

✓ Same CLI flags ( -e / --epochs, -c / --comments, --plots )
✓ Saves the **best checkpoint** (highest val ROC-AUC) under artifacts/checkpoints/
✓ Logs **accuracy, precision, recall, F1, ROC-AUC**
✓ Exports **confusion-matrix** and **ROC curve** PNGs to artifacts/plots/

Requirements
------------
pip install torch torchaudio transformers scikit-learn matplotlib
"""

from __future__ import annotations
import argparse, os, random, json, inspect
from pathlib import Path
from datetime import datetime

import numpy as np
import torch, torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split

from transformers.models.wav2vec2 import (
    Wav2Vec2Processor, Wav2Vec2ForSequenceClassification,
)
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.trainer_utils import set_seed
from transformers.trainer_callback import TrainerCallback
from transformers.optimization import get_scheduler
import inspect, transformers

# ---------- FOLDER CONFIG ----------
DATA_ROOT      = Path("data-source/audio")
CHECKPOINT_DIR = Path("artifacts/checkpoints")
PLOT_DIR       = Path("artifacts/plots")
STATS_DIR      = Path("artifacts/stats")
for d in (CHECKPOINT_DIR, PLOT_DIR, STATS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- AUDIO CONFIG ----------
SAMPLE_RATE = 16_000            # Wav2Vec2 expects 16 kHz
MAX_SECONDS = 4.0               # pad / truncate (≈64 k frames)
MODEL_ID    = "facebook/wav2vec2-base-960h"
FREEZE_PCT  = 0.8               # freeze bottom 80 % encoder layers
BATCH_SIZE  = 4
RANDOM_SEED = 42
set_seed(RANDOM_SEED)

# ---------- DATASET ----------
class PDVoiceDataset(Dataset):
    def __init__(self, files: list[Path], processor: Wav2Vec2Processor, augment=False):
        self.files, self.processor, self.augment = files, processor, augment
        self.labels = [0 if f.parent.name.startswith("HC") else 1 for f in files]
        self.max_len = int(SAMPLE_RATE * MAX_SECONDS)

    def _load_audio(self, wav_path: Path):
        wav, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(dim=0)            # mono

        # mild augmentation
        if self.augment:
            if random.random() < 0.3:        # gain
                wav = wav * random.uniform(0.9, 1.1)
            if random.random() < 0.3:        # white noise
                wav = wav + 0.005 * torch.randn_like(wav)

        # pad / truncate
        if len(wav) < self.max_len:
            wav = torch.nn.functional.pad(wav, (0, self.max_len - len(wav)))
        else:
            wav = wav[: self.max_len]
        return wav

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        wav = self._load_audio(self.files[idx])
        length = len(wav)                            # after pad / truncate
        mask   = torch.zeros(self.max_len, dtype=torch.long)
        mask[: length] = 1                           # 1 = real audio, 0 = padding

        inputs = self.processor(
            wav.numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",                    # keep fixed length
            max_length=self.max_len,
        )

        item = {
            "input_values"   : inputs["input_values"].squeeze(0),  # (T,)
            "attention_mask" : mask,                               # (T,)
            "labels"         : torch.tensor(self.labels[idx],
                                            dtype=torch.long),
        }
        return item

# ---------- CALLBACKS ----------    
class UnfreezeTopLayersCallback(TrainerCallback):
    """
    Un-freeze the last `n_layers` transformer blocks at a chosen epoch and
    give them their own learning-rate.

    Args
    ----
    unfreeze_at_epoch : int  (epoch number, 0-based → 3 means “after epoch 3”)
    n_layers          : int  (how many top encoder layers to un-freeze)
    base_lr           : float  (LR for the new param group; default =
                                TrainingArguments.learning_rate)
    """
    def __init__(self,
                 unfreeze_at_epoch: int = 3,
                 n_layers: int = 4,
                 base_lr: float | None = None):
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.n_layers = n_layers
        self.base_lr = base_lr        # if None we’ll read it from args
        self.done = False

    # ----------------------------------------------
    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.done or state.epoch < self.unfreeze_at_epoch:
            return

        model     = kwargs["model"]
        optimizer = kwargs["optimizer"]

        # 1 ─ un-freeze the desired encoder layers
        new_params = []
        for layer in model.wav2vec2.encoder.layers[-self.n_layers:]:
            for p in layer.parameters():
                if not p.requires_grad:          # was frozen
                    p.requires_grad = True
                    new_params.append(p)

        # 2 ─ add *one* new param-group with its own LR
        lr_new  = self.base_lr or args.learning_rate
        optimizer.add_param_group({
            "params"      : new_params,
            "lr"          : lr_new,
            "weight_decay": 0.01,
        })

        trainable = sum(p.requires_grad for p in model.parameters())
        print(f"[Callback] Un-froze top {self.n_layers} layers at epoch "
              f"{int(state.epoch)} → trainable tensors {trainable}; "
              f"head-LR {optimizer.param_groups[0]['lr']:.1e}, "
              f"encoder-LR {lr_new:.1e}")

        self.done = True

# ---------- METRICS ----------
def compute_metrics(pred):
    logits = pred.predictions
    y_prob = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
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
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
    }

# ---------- PLOTS ----------
def plot_confusion_matrix(cm, out_path):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["HC", "PD"])
    plt.yticks(ticks, ["HC", "PD"])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(int(cm[i, j])),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_roc(y_true, y_prob, out_path):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        return
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve (AUC = {auc:.3f})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------- UTILS ----------
def freeze_bottom_layers(model, pct):
    total = len(model.wav2vec2.encoder.layers)
    for i, layer in enumerate(model.wav2vec2.encoder.layers):
        if i < int(total * pct):
            for p in layer.parameters():
                p.requires_grad = False

def list_audio_files() -> list[Path]:
    files = []
    for sub in ("HC_AH", "PD_AH"):
        files.extend((DATA_ROOT / sub).glob("*.wav"))
    return sorted(files, key=lambda p: p.name)

# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune Wav2Vec 2.0 for PD detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("-e", "--epochs", type=int, default=20)
    ap.add_argument("-c", "--comments", type=str, default=None)
    ap.add_argument("--plots", action="store_true", help="Save confusion-matrix & ROC plots")
    ap.add_argument("-l", "--layers", type=int, default=6, choices=[2,6],
                    help="Number of transformer layers to unfreeze during fine-tuning")
    ap.add_argument("-u", "--unfreeze", type=int, default=3, help="Epoch to unfreeze layers")
    args = ap.parse_args()

    print(f"[TIME] {datetime.now():%Y-%m-%d %H:%M:%S}")
    if args.comments:
        print("[COMMENT]", args.comments)
    print("[INFO] Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using Wav2Vec2 model:", MODEL_ID)
    print("--------------ENVIRONMENT-----------------")
    print("🤗 Transformers version :", transformers.__version__)
    print("package imported from  :", os.path.dirname(transformers.__file__))
    print("TrainingArguments file :", inspect.getfile(TrainingArguments))
    print("------------------------------------------")

    # ----- data split -----
    all_files = list_audio_files()
    labels = [0 if f.parent.name.startswith("HC") else 1 for f in all_files]
    train_f, val_f = train_test_split(
        all_files, test_size=0.3,
        stratify=labels, random_state=RANDOM_SEED
    )

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, do_normalize=False)
    train_ds = PDVoiceDataset(train_f, processor, augment=True)
    val_ds   = PDVoiceDataset(val_f,   processor, augment=False)

    learning_rate = 5e-4 if torch.cuda.is_available() else 1e-5

    # ----- model -----
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2,
        attention_dropout=0.05, hidden_dropout=0.05,
        gradient_checkpointing=True if torch.cuda.is_available() else False
    )
    # freeze_bottom_layers(model, 1.0)#FREEZE_PCT)

    # Freeze Feature Extractor
    for p in model.wav2vec2.feature_extractor.parameters():
        p.requires_grad = False

    # Freeze *everything* …
    for p in model.wav2vec2.parameters():
        p.requires_grad = False

    # … then un-freeze only the small projector + classifier head
    for n, p in model.named_parameters():
        if any(k in n for k in ("projector", "classifier")):
            p.requires_grad = True

    # override the default AdamW optimizer
    # (which uses all model parameters) to only train the head
    # head_params = [p for p in model.parameters() if p.requires_grad]
    # optim = torch.optim.AdamW(head_params, lr=learning_rate, weight_decay=0.01)

    # ─────────────────────────── debug  ─────────────────────────
    batch = next(iter(DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)))
    for k, v in batch.items():
        print(k, v.shape, v.dtype, v.min().item(), v.max().item())

    with torch.no_grad():
        out = model(input_values=batch["input_values"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
    print("forward  →  loss:", out.loss)
    print("any NaN?", torch.isnan(out.logits).any().item())
    # ───────────────────────────────────────────────────────────────────────────

    training_args = TrainingArguments(
        output_dir                  = str(CHECKPOINT_DIR),
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate               = learning_rate,
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
        model                       = model,
        args                        = training_args,
        train_dataset               = train_ds,
        eval_dataset                = val_ds,
        processing_class            = processor,
        compute_metrics             = compute_metrics,
        # optimizers                  = (optim, None),
        callbacks                   = [UnfreezeTopLayersCallback(unfreeze_at_epoch=args.unfreeze,
                                        n_layers=args.layers,
                                        base_lr=training_args.learning_rate)],
    )

    trainer.train()

    # ----- evaluation -----
    preds = trainer.predict(val_ds)
    metrics = compute_metrics(preds)
    y_prob = torch.softmax(torch.tensor(preds.predictions), dim=-1)[:, 1].numpy()
    y_true = preds.label_ids
    if y_true is not None:
        cm = confusion_matrix(y_true, (y_prob > 0.5).astype(int))
    else:
        cm = np.zeros((2, 2))  # fallback empty confusion matrix
    metrics["confusion_matrix"] = cm.tolist()

    # save best checkpoint path + metrics
    best_ckpt = Path(trainer.state.best_model_checkpoint or CHECKPOINT_DIR)
    meta_path = best_ckpt / "eval_metrics.json"
    with open(meta_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[CHECKPOINT] Best model @ {best_ckpt}")
    print("--- FINAL VALIDATION METRICS ---")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k:10}: {v}")

    # plots
    if args.plots:
        plot_confusion_matrix(cm, STATS_DIR / "confusion_matrix.png")
        plot_roc(y_true, y_prob, STATS_DIR / "roc_curve.png")
        print("Confusion-matrix and ROC curve saved to artifacts/plots/")
    
    print("[INFO] Training complete.\n\n")

if __name__ == "__main__":
    main()
