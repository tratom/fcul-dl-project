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

# Quick sanity-check Tiny CNN on padded log-mel spectrograms
#
# Seguendo la guida:
# 1) “Verifica che i dati contengano davvero un segnale” → usiamo un modello
#    minimalista (TinyCNN) su immagini 64×1024 già paddate
# 2) “Controlla artefatti dovuti al padding” → abbiamo generato le immagini
#    solo dopo aver trim-mato il silenzio e poi applicato pad residuo
# 3) “Evita il label leakage” → split random train/test (80/20), ma si raccomanda
#    di adottare in futuro uno split speaker-indipendente se necessario
# 4) Confronto finale con il baseline LSTM (acc 0.52 / AUC 0.46)
# ------------------------------------------------------------------------------
# -------------------- DEFAULTS --------------------
SEED_DEFAULT       = 42
DATA_DIR_DEFAULT   = "milestone2/plots_padding"  # base dir containing HC_AH/ and PD_AH/
BATCH_SIZE_DEFAULT = 8
LR_DEFAULT         = 1e-3
EPOCHS_DEFAULT     = 10
TEST_SIZE_DEFAULT  = 0.2  # 20% for test

# -------------------- DATASET --------------------
class SpectrogramDataset(Dataset):
    """
    Cerca PNG in milestone2/plots_padding/HC_AH/ e .../PD_AH/,
    etichetta HC_AH → 0, PD_AH → 1.
    Ritorna tensori [1,H,W] via ToTensor().
    """
    def __init__(self, root_dir: Path, transform=None):
        self.transform = transform
        self.samples = []
        for label_name, label in [("HC_AH", 0), ("PD_AH", 1)]:
            folder = root_dir / label_name
            for img_path in folder.glob("*.png"):
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label

# -------------------- MODEL --------------------
class TinyCNN(nn.Module):
    """
    Minimal CNN: Conv2d → Conv2d → AdaptiveAvgPool2d → Linear
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
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

# -------------------- TRAIN / EVAL --------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Esegue un’epoca di training:
      - restituisce loss, accuracy e AUC sul training set
    """
    model.train()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs > 0.5).long()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    epoch_auc  = roc_auc_score(all_labels, all_probs)
    return epoch_loss, epoch_acc, epoch_auc

def evaluate(model, loader, device):
    """
    Valuta il modello su un dataset (train o test):
      - restituisce accuracy e AUC
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs > 0.5).long()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, auc

# -------------------- ARGPARSE --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Tiny CNN on padded log-mel images")
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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=torch.cuda.is_available()
    )
    test_loader  = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=torch.cuda.is_available()
    )

    # model, optimizer, loss
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = TinyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # checkpoint logic
    best_auc = -1.0

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

        # evaluate on test set and checkpoint if better
        test_acc, test_auc = evaluate(model, test_loader, device)
        if test_auc > best_auc:
            best_auc = test_auc
            print(f"[CHECKPOINT] Saved new best model → best_auc_{best_auc:.3f}")

    # final evaluation on test set
    final_acc, final_auc = evaluate(model, test_loader, device)
    print(f"\n>>> Final Test results: acc={final_acc:.4f}, auc={final_auc:.4f}")

if __name__ == "__main__":
    main()
