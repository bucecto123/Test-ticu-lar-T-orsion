# dataCleaning/evaluation.py
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Gradio
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from algebraic_engine import AlgebraicMonoidEngine
from neuro_symbolic_graph import BASE_THRESHOLD

# Load math artifacts once at module import
_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'subspace_models')
_p_u = np.load(os.path.join(_MODELS_DIR, 'p_u_matrix.npy'))
_scaler = joblib.load(os.path.join(_MODELS_DIR, 'geometric_scaler.joblib'))
_engine = AlgebraicMonoidEngine()


def _predict_flow(row_dict: dict) -> tuple:
    """
    Runs both deterministic sensors on a single flow dict.
    Returns (continuous_anomaly: bool, algebraic_anomaly: bool).
    """
    # --- Continuous sensor ---
    expected_features = _scaler.feature_names_in_
    raw = {feat: [row_dict.get(feat, 0.0)] for feat in expected_features}
    x_df = pd.DataFrame(raw)
    x_scaled = _scaler.transform(x_df)
    x_proj = np.dot(x_scaled, np.dot(_p_u, _p_u.T))
    w = x_scaled - x_proj
    residual = float(np.linalg.norm(w))
    continuous_anomaly = bool(residual > BASE_THRESHOLD)

    # --- Discrete sensor ---
    result = _engine.evaluate_state_loop(row_dict)
    algebraic_anomaly = bool(result["algebraic_anomaly"])

    return continuous_anomaly, algebraic_anomaly


def run_evaluation_pipeline(csv_path: str, sample_size: int = 500):
    """
    Evaluates both deterministic sensors against ground truth labels.

    Args:
        csv_path: Path to a CIC-IDS2017 format CSV.
        sample_size: Total number of flows to evaluate (spread across classes).

    Returns:
        (fig, metrics_df): matplotlib Figure (confusion matrix) and DataFrame (per-class metrics).
    """
    # Load and clean
    df = pd.read_csv(csv_path, nrows=sample_size * 10)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Sample evenly across classes
    classes = df['Label'].unique()
    per_class = max(1, sample_size // len(classes))
    sampled = pd.concat(
        [df[df['Label'] == lbl].head(per_class) for lbl in classes]
    ).reset_index(drop=True)

    y_true_binary = []
    y_pred_binary = []

    for _, row in sampled.iterrows():
        row_dict = row.to_dict()
        cont_anom, alg_anom = _predict_flow(row_dict)
        predicted_attack = cont_anom or alg_anom
        true_label = row['Label']

        y_true_binary.append("BENIGN" if true_label == "BENIGN" else "ATTACK")
        y_pred_binary.append("ATTACK" if predicted_attack else "BENIGN")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=["BENIGN", "ATTACK"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["BENIGN", "ATTACK"])
    ax.set_yticklabels(["BENIGN", "ATTACK"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Subspace Aegis IDS")
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14, color=color)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    # --- Per-class metrics table ---
    report = classification_report(
        y_true_binary, y_pred_binary, output_dict=True, zero_division=0
    )
    metrics_df = pd.DataFrame(report).T.round(3)
    metrics_df = metrics_df[
        metrics_df.index.isin(["BENIGN", "ATTACK", "accuracy", "macro avg", "weighted avg"])
    ]

    return fig, metrics_df
