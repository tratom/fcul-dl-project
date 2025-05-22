from __future__ import annotations
import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold

# -------------------- CONFIG --------------------

DATA_ROOT      = Path("data-source/audio")
AUG_DIR        = Path("artifacts/augmented_audio")
PLOT_DIR       = Path("milestone2/plots_augmented")
CHECKPOINT_DIR = Path("milestone2/checkpoints_augmented")
STATS_DIR      = Path("milestone2/stats_augmented")
for d in (CHECKPOINT_DIR, PLOT_DIR, STATS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16_000
N_MELS      = 64
HOP_LENGTH  = 160
WIN_LENGTH  = 400
FMIN        = 50
FMAX        = 4_000
MAX_FRAMES  = 1_024

RANDOM_SEED = 42
BATCH_SIZE  = 8
NUM_WORKERS = os.cpu_count() or 2
EPSILON     = 0.1