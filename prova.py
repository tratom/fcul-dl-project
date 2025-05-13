#!/usr/bin/env python
"""
Milestone 1 - Early Parkinson's Detection Using Speech Analysis
Using Wav2Vec2.0 backbone + classification head
------------------------------------------------
This script replaces the log-mel + LSTM pipeline with a HuggingFace Wav2Vec2-based
sequence classifier. It:
  • Loads raw waveforms, resampling to 16kHz if needed
  • Generates attention masks for padded inputs
  • Uses a pretrained Wav2Vec2 model as feature extractor
  • Fine-tunes a lightweight classification head
  • Saves best checkpoint, computes accuracy, precision, recall, F1, ROC-AUC
  • Exports confusion-matrix and ROC curve under `artifacts/plots/`

Usage: python milestone1_wav2vec2.py [options]
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import soundfile as sf
import librosa

# Directories & constants
DATA_ROOT = Path("data-source/audio")
CHECKPOINT_DIR = Path("artifacts/checkpoints")
PLOT_DIR = Path("artifacts/plots")
STATS_DIR = Path("artifacts/stats")
SAMPLE_RATE = 16_000
RANDOM_SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count() or 2
MODEL_NAME = "facebook/wav2vec2-base"

# Ensure reproducibility
torch.manual_seed(RANDOM_SEED)

# ---------- Data utilities ----------

def list_wav_files() -> Tuple[List[Path], List[int]]:
    """Return list of wav file paths and labels (0=HC, 1=PD)."""
    files: list[Path] = []
    labels: list[int] = []
    for label_name, label in [("HC_AH", 0), ("PD_AH", 1)]:
        for wav in (DATA_ROOT / label_name).glob("*.wav"):
            files.append(wav)
            labels.append(label)
    return files, labels

class ParkinsonWaveformDataset(Dataset):
    def __init__(self, files: List[Path], processor: Wav2Vec2Processor):
        self.files = files
        self.processor = processor
        self.labels = [0 if f.parent.name.startswith("HC_") else 1 for f in files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        speech, sr = sf.read(path, dtype="float32")
        if sr != SAMPLE_RATE:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=SAMPLE_RATE)
        # Processor returns only input_values for single samples without padding
        inputs = self.processor(
            speech,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.squeeze(0)
        # create attention mask (ones for actual samples)
        attention_mask = torch.ones(input_values.shape, dtype=torch.long)
        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# Collate function for variable-length inputs

def collate_fn(batch: list[dict]) -> dict:
    input_vals = [b["input_values"] for b in batch]
    masks = [b["attention_mask"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    input_vals = nn.utils.rnn.pad_sequence(input_vals, batch_first=True, padding_value=0.0)
    masks = nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    return {"input_values": input_vals, "attention_mask": masks, "labels": labels}

# ---------- Model training & evaluation ----------

def step_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train() if train else model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        iv = batch["input_values"].to(device)
        am = batch["attention_mask"].to(device)
        lb = batch["labels"].to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(iv, attention_mask=am)
        logits = outputs.logits
        loss = criterion(logits, lb)
        if train:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * lb.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == lb).sum().item()
        total += lb.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, device="cpu") -> dict:
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch in loader:
            iv = batch["input_values"].to(device)
            am = batch["attention_mask"].to(device)
            lb = batch["labels"].to(device)
            outputs = model(iv, attention_mask=am)
            logits_list.append(outputs.logits.cpu())
            labels_list.append(lb.cpu())
    logits = torch.cat(logits_list).numpy()
    y_true = torch.cat(labels_list).numpy()
    probs = torch.softmax(logits, axis=1)[:, 1]
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, probs)
    except ValueError:
        roc_auc = float("nan")
    cm = confusion_matrix(y_true, preds)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc_auc, "cm": cm, "probs": probs, "y_true": y_true}

# ---------- Plotting ----------

def plot_confusion_matrix(cm: np.ndarray, path: Path):
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["HC", "PD"])
    plt.yticks(ticks, ["HC", "PD"])
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, probs: np.ndarray, path: Path):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve (AUC = {roc_auc_score(y_true, probs):.3f})")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        prog="milestone1_wav2vec2.py",
        usage="%(prog)s [options]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Early Parkinson's Detection Using Wav2Vec2.0 - Milestone 1",
    )
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-c", "--comments", type=str, default=None, help="Optional log comment")
    args = parser.parse_args()

    print(f"EXECUTION TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.comments:
        print("[COMMENT]", args.comments)

    all_files, all_labels = list_wav_files()
    train_files, val_files, _, _ = train_test_split(all_files, all_labels, test_size=0.3, stratify=all_labels, random_state=RANDOM_SEED)

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    train_ds = ParkinsonWaveformDataset(train_files, processor)
    val_ds = ParkinsonWaveformDataset(val_files, processor)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True).to(device)
    for param in model.wav2vec2.feature_extractor.parameters(): param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    best_roc = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = step_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = step_epoch(model, val_dl, criterion, None, device)
        print(f"Epoch {epoch:02d} | train acc {tr_acc:.3f} | val acc {val_acc:.3f} | train loss {tr_loss:.3f} | val loss {val_loss:.3f}")
        val_metrics = evaluate(model, val_dl, device)
        if not np.isnan(val_metrics["roc_auc"]) and val_metrics["roc_auc"] > best_roc:
            best_roc = val_metrics["roc_auc"]
            ckpt_path = CHECKPOINT_DIR / f"best_auc_{best_roc:.3f}.pt"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metrics": val_metrics}, ckpt_path)
            print(f"[CHECKPOINT] Saved new best model → {ckpt_path.name}")

    final_metrics = evaluate(model, val_dl, device)
    print("--- FINAL VALIDATION METRICS ---")
    for k, v in final_metrics.items():
        if k not in ("probs", "y_true"):
            print(f"{k:10}: {v}")
    plot_confusion_matrix(final_metrics["cm"], STATS_DIR / "confusion_matrix.png")
    plot_roc_curve(final_metrics["y_true"], final_metrics["probs"], STATS_DIR / "roc_curve.png")
    print("Confusion matrix and ROC curve saved to artifacts/plots/")
    print("[INFO] Finished")

if __name__ == "__main__":
    main()
