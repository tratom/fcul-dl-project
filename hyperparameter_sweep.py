import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from milestone1_skeleton import (
    LSTMAudioClassifier,
    cache_all,
    CACHE_DIR,
    ParkinsonDataset,
    step_epoch as train_one_epoch,
    evaluate,
)

# Helper to load datasets

def get_train_val_datasets(test_size: float = 0.3, random_state: int = 42):
    # Ensure spectrograms are cached
    cache_all()
    # Gather precomputed specs
    all_specs = sorted(CACHE_DIR.glob("*.npy"))
    labels = [0 if f.name.startswith("HC_") else 1 for f in all_specs]
    train_files, val_files = train_test_split(
        all_specs,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    train_ds = ParkinsonDataset(train_files)
    val_ds = ParkinsonDataset(val_files)
    return train_ds, val_ds


def run_sweep():
    # Hyperparameter grid
    grid = {
        'hidden_size': [128, 256, 512],
        'num_layers': [1, 2, 3],
        'dropout': [0.3, 0.5, 0.7],
        'bidirectional': [False, True],
    }
    keys, values = zip(*grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Load datasets
    train_ds, val_ds = get_train_val_datasets()
    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []

    for idx, params in enumerate(param_combinations, 1):
        print(f"Running {idx}/{len(param_combinations)}: {params}")
        model = LSTMAudioClassifier(
            n_mels=train_ds[0][0].shape[1],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            bidirectional=params['bidirectional'],
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val_acc = 0.0
        for epoch in range(1, 11):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            val_acc = val_metrics['acc']
            print(f" Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
            best_val_acc = max(best_val_acc, val_acc)

        results.append({**params, 'best_val_acc': best_val_acc})

    # Report top 5
    top5 = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)[:5]
    print("\nTop 5 configs:")
    for r in top5:
        print(r)

    # Save full results
    import pandas as pd
    pd.DataFrame(results).to_csv('hyperparam_results.csv', index=False)
    print("Saved results to hyperparam_results.csv")


if __name__ == '__main__':
    run_sweep()