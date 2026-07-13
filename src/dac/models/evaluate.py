"""Model performance evaluation: threshold-free and threshold-based metrics,
plus diagnostic plots (ROC, PR curve, calibration, confusion matrix).

Includes PR-AUC and Brier score alongside AUC-ROC/F1 specifically because
Home Credit's ~8% default rate makes AUC-ROC alone potentially misleading
under class imbalance (flagged in the synopsis reviewer critique).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)


def ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between the cumulative
    distributions of predicted scores for the positive and negative classes.
    Standard credit-scoring metric alongside AUC-ROC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(np.abs(tpr - fpr)))


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "ks_statistic": ks_statistic(y_true, y_proba),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "threshold": threshold,
    }


def plot_evaluation_suite(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    figures_dir: Path,
    threshold: float = 0.5,
) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[0, 0].plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_proba):.3f}", color="#4C72B0")
    axes[0, 0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0, 0].set_title(f"{model_name}: ROC curve")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].legend()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    axes[0, 1].plot(recall, precision, label=f"PR-AUC={average_precision_score(y_true, y_proba):.3f}", color="#C44E52")
    axes[0, 1].set_title(f"{model_name}: Precision-Recall curve")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend()

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    axes[1, 0].plot(prob_pred, prob_true, marker="o", color="#55A868")
    axes[1, 0].plot([0, 1], [0, 1], "--", color="grey")
    axes[1, 0].set_title(f"{model_name}: Calibration curve")
    axes[1, 0].set_xlabel("Mean predicted probability")
    axes[1, 0].set_ylabel("Fraction of positives")

    # Confusion matrix
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    im = axes[1, 1].imshow(cm, cmap="Blues")
    axes[1, 1].set_title(f"{model_name}: Confusion matrix (thr={threshold})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 1].text(j, i, str(cm[i, j]), ha="center", va="center")
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("Actual")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

    fig.tight_layout()
    fig.savefig(figures_dir / f"{model_name}_evaluation_suite.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def compare_models(results: dict[str, dict], metrics_dir: Path) -> pd.DataFrame:
    """results: {model_name: metrics_dict}. Writes a comparison table + JSON."""
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results).T
    df.to_csv(metrics_dir / "model_comparison.csv")
    (metrics_dir / "model_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5))
    df[["roc_auc", "pr_auc", "f1", "ks_statistic"]].plot(kind="bar", ax=ax)
    ax.set_title("Model comparison")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(metrics_dir / "model_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    logger.info("Model comparison:\n%s", df.round(4).to_string())
    return df
