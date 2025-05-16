import os
from pathlib import Path
import argparse

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, roc_auc_score

# -------------------- DEFAULTS --------------------
SEED_DEFAULT       = 42
DATA_DIR_DEFAULT   = "artifacts/plots"
BATCH_SIZE_DEFAULT = 8
LR_DEFAULT         = 1e-3
EPOCHS_DEFAULT     = 10
TEST_SIZE_DEFAULT  = 0.2  # 20% for test

# -------------------- DATASET --------------------
class SpectrogramDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.transform = transform
        self.samples = []
        for img_path in root_dir.glob("*.png"):
            if img_path.name.startswith("PD_"):
                label = 1
            elif img_path.name.startswith("HC_"):
                label = 0
            else:
                continue
            self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label

# -------------------- MODEL (increased capacity) --------------------
class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            # increased channels + extra conv block
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# -------------------- TRAIN / EVAL UTILITIES --------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds  = []
    all_probs  = []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(logits, dim=1)[:,1]
        preds = (probs > 0.5).long()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    epoch_auc  = roc_auc_score(all_labels, all_probs)
    return epoch_loss, epoch_acc, epoch_auc

def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)[:,1]
            preds  = (probs > 0.5).long()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, auc

# -------------------- ARGPARSE --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Tiny CNN on log-mel spectrogram images")
    p.add_argument("--data-dir",  type=str,   default=DATA_DIR_DEFAULT)
    p.add_argument("--batch-size",type=int,   default=BATCH_SIZE_DEFAULT)
    p.add_argument("--lr",        type=float, default=LR_DEFAULT)
    p.add_argument("--epochs",    type=int,   default=EPOCHS_DEFAULT)
    p.add_argument("--test-size", type=float, default=TEST_SIZE_DEFAULT)
    p.add_argument("--seed",      type=int,   default=SEED_DEFAULT)
    return p.parse_args()

# -------------------- MAIN --------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # transforms + dataset
    transform = transforms.Compose([transforms.ToTensor()])
    full_ds = SpectrogramDataset(Path(args.data_dir), transform=transform)

    # split train/test
    N      = len(full_ds)
    n_test = int(N * args.test_size)
    n_train= N - n_test
    train_ds, test_ds = random_split(
        full_ds,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=os.cpu_count(),
                              pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=os.cpu_count(),
                              pin_memory=torch.cuda.is_available())

    # model, optimizer, loss
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = TinyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # training loop
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"train_auc={train_auc:.4f}"
        )

    # final evaluation on test set
    test_acc, test_auc = evaluate(model, test_loader, device)
    print(f"\n>>> Test results: acc={test_acc:.4f}, auc={test_auc:.4f}")

if __name__ == "__main__":
    main()
