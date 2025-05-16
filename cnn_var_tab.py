import argparse
import pandas as pd
import numpy as np
import joblib
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


class CSVDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(0)  # (1, n_features)
        y = self.y[idx]
        return x, y


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class FeatureCNN(nn.Module):
    def __init__(self, input_length, base_filters=32, dropout=0.3):
        super().__init__()
        f = base_filters
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, f, kernel_size=3, padding=1),
            nn.BatchNorm1d(f), nn.ReLU(inplace=True),
            ResidualBlock(f, dropout),
            nn.MaxPool1d(2), nn.Dropout(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(f, f*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(f*2), nn.ReLU(inplace=True),
            ResidualBlock(f*2, dropout),
            nn.MaxPool1d(2), nn.Dropout(dropout)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(f*2, f*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(f*4), nn.ReLU(inplace=True),
            ResidualBlock(f*4, dropout),
            nn.MaxPool1d(2), nn.Dropout(dropout)
        )
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(f*4, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_avg(x).squeeze(-1)
        return self.fc(x)


def cross_validate(csv_path, n_splits=5, random_seed=42):
    # Load data
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    df['Label_enc'] = le.fit_transform(df['Label'])
    df['Sex_enc'] = (df['Sex'] == 'F').astype(int)
    features = [
        'meanF0','stdevF0','hnr','localJitter','localabsoluteJitter',
        'rapJitter','ppq5Jitter','ddpJitter','localShimmer','localdbShimmer',
        'apq3Shimmer','apq5Shimmer','apq11Shimmer','ddaShimmer',
        'Age','Sex_enc'
    ]
    X = df[features].values
    y = df['Label_enc'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_metrics = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Balance classes with sampler
        class_counts = np.bincount(y_train.astype(int))
        weights = 1.0 / class_counts
        sample_weights = weights[y_train.astype(int)]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_ds = CSVDataset(X_train, y_train)
        val_ds = CSVDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        model = FeatureCNN(input_length=X.shape[1]).to(device)
        optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        criterion = BCEWithLogitsLoss()

        # Train
        epochs = 30
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Validate
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                probs = torch.sigmoid(model(xb)).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                y_true.extend(yb.numpy().astype(int))
                y_pred.extend(preds.tolist())

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        fold_metrics.append((acc, prec, rec, f1))
        print(f"Fold {fold}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

    # Aggregate
    fold_metrics = np.array(fold_metrics)
    mean_metrics = fold_metrics.mean(axis=0)
    std_metrics = fold_metrics.std(axis=0)
    print("\nCross-Validation Results:")
    print(f"Accuracy: {mean_metrics[0]:.3f} ± {std_metrics[0]:.3f}")
    print(f"Precision: {mean_metrics[1]:.3f} ± {std_metrics[1]:.3f}")
    print(f"Recall: {mean_metrics[2]:.3f} ± {std_metrics[2]:.3f}")
    print(f"F1 Score: {mean_metrics[3]:.3f} ± {std_metrics[3]:.3f}\n")
    print("Average Classification Report (per fold):")
    # Optionally print detailed per-class metrics aggregated separately


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="artifacts/vsp_all_features_with_demographics.csv", help="Path to CSV file")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()
    cross_validate(args.csv, n_splits=args.folds)