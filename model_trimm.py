import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns

# === Dataset ===
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Modello ===
class AudioClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

#'trimmed_audio', 'trimmed_audio_features.csv'
# === Training e Valutazione ===
def main():
    # Carica features
    data_path = os.path.join('trimmed_audio', 'trimmed_audio_features.csv')
    df = pd.read_csv(data_path)

    # Rimuovi righe con valori NaN
    df = df.dropna()

    # Mappa label a numeri: HC->0, PD->1
    #label
    #PD
    df['label_bin'] = df['label'].map({'HC': 0, 'PD': 1})

    # Separa X e y
    #'wav_file', 'label', 'wav_path', 'label_bin'
    #'Sex' 'Age'
    drop_cols = ['wav_file', 'label', 'wav_path', 'label_bin']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values
    y = df['label_bin'].values

    # Standardizzazione delle feature
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split train/val/test (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # DataLoader
    train_ds = AudioDataset(X_train, y_train)
    val_ds = AudioDataset(X_val, y_val)
    test_ds = AudioDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    # Setup modello
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X.shape[1]
    model = AudioClassifier(input_dim).to(device)
    criterion = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    n_epochs = 50
    train_losses, val_losses = [], []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                val_loss += criterion(outputs, y_batch).item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{n_epochs}  Train Loss: {train_losses[-1]:.4f}  Val Loss: {val_losses[-1]:.4f}")

    # Salva modello e curve di loss
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('models', 'audio_classifier.pt'))
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('models', 'loss_curve.png'))

    # Test finale
    model.eval()
    y_preds, y_trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            preds = torch.round(torch.sigmoid(outputs))
            y_preds.extend(preds.cpu().numpy())
            y_trues.extend(y_batch.numpy())

    # Metriche e report
    acc = accuracy_score(y_trues, y_preds) * 100
    print(f"Test Accuracy: {acc:.2f}%")
    print(classification_report(y_trues, y_preds, target_names=['HC', 'PD']))
    cm = confusion_matrix(y_trues, y_preds, labels=[0,1])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['HC','PD'], yticklabels=['HC','PD'])
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join('models', 'confusion_matrix.png'))

if __name__ == '__main__':
    main()
