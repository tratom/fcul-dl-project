import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Percorsi dei file
vggish_path = "C:/Users/cri_t/OneDrive/Desktop/Lisboa_esami/data_deep/ROC/vggish_metrics.json"
lstm_path = "C:/Users/cri_t/OneDrive/Desktop/Lisboa_esami/data_deep/ROC/lstm_metrics.json"

plt.figure(figsize=(8, 6))

# --- VGGish: ha già fpr e tpr nel file ---
with open(vggish_path, "r") as f:
    vggish = json.load(f)

plt.plot(
    vggish["fpr"],
    vggish["tpr"],
    label=f"VGGish (AUC = {vggish['roc_auc']:.2f})",
    linewidth=2
)

# --- LSTM: ha y_true e y_score, calcoliamo FPR/TPR ---
with open(lstm_path, "r") as f:
    lstm = json.load(f)

fpr_lstm, tpr_lstm, _ = roc_curve(lstm["y_true"], lstm["y_score"])
roc_auc_lstm = lstm["roc_auc"]

plt.plot(
    fpr_lstm,
    tpr_lstm,
    label=f"LSTM (AUC = {roc_auc_lstm:.2f})",
    linewidth=2
)

# --- Linea random + formato grafico ---
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
