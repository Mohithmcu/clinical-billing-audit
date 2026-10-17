"""Evaluation metrics for the FWA mismatch detection pipeline.

Provides precision, recall, F1, PR-AUC, and ROC-AUC calculations plus
visualisation helpers for precision-recall curves and confusion matrices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)

# Guard matplotlib imports for headless environments
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def compute_binary_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    y_score: np.ndarray | list | None = None,
) -> dict[str, float]:
    """Compute binary classification metrics for mismatch detection.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (1 = mismatch, 0 = correct).
    y_pred : array-like
        Predicted binary labels.
    y_score : array-like | None
        Predicted scores/probabilities for the positive class.

    Returns
    -------
    dict[str, float]
        Dictionary of metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["true_negatives"] = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    metrics["false_positives"] = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    metrics["false_negatives"] = int(cm[1, 0]) if cm.shape == (2, 2) else 0
    metrics["true_positives"] = int(cm[1, 1]) if cm.shape == (2, 2) else 0

    # AUC metrics (require scores)
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = 0.0
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            metrics["pr_auc"] = 0.0

    return metrics


def compute_multiclass_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute multi-class classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    class_names : list[str] | None
        Class names for the report.

    Returns
    -------
    dict
        Classification report + overall metrics.
    """
    from sklearn.metrics import classification_report, f1_score

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report["overall_f1_weighted"] = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    report["overall_f1_macro"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    return report


def compute_threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep thresholds and compute precision/recall at each.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth.
    y_score : np.ndarray
        Continuous scores.
    thresholds : np.ndarray | None
        Thresholds to evaluate (default: 50 evenly spaced).

    Returns
    -------
    pd.DataFrame
        Precision, recall, F1 at each threshold.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    if thresholds is None:
        thresholds = np.linspace(0, 1, 51)

    rows = []
    for t in thresholds:
        preds = (y_score >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "n_flagged": int(preds.sum()),
            "flag_rate": float(preds.mean()),
        })

    return pd.DataFrame(rows)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: str | Path | None = None,
    title: str = "Precision-Recall Curve — Mismatch Detection",
) -> None:
    """Plot precision-recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth.
    y_score : np.ndarray
        Continuous scores.
    output_path : str | Path | None
        If provided, saves the figure.
    title : str
        Plot title.
    """
    if not HAS_PLOT:
        logger.warning("Matplotlib not available — skipping plot.")
        return

    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#2563eb", linewidth=2,
            label=f"AP = {ap:.3f}")
    ax.fill_between(recall, precision, alpha=0.1, color="#2563eb")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("PR curve saved to %s", output_path)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    output_path: str | Path | None = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot confusion matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted labels.
    class_names : list[str] | None
        Class names for axis labels.
    output_path : str | Path | None
        If provided, saves the figure.
    title : str
        Plot title.
    """
    if not HAS_PLOT:
        logger.warning("Matplotlib not available — skipping plot.")
        return

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names or "auto",
        yticklabels=class_names or "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", output_path)
    plt.close(fig)


def generate_evaluation_report(
    metrics: dict[str, float],
    threshold_df: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Generate a markdown evaluation report.

    Parameters
    ----------
    metrics : dict[str, float]
        Computed metrics.
    threshold_df : pd.DataFrame | None
        Threshold sweep results.
    output_path : str | Path | None
        If provided, saves the report.

    Returns
    -------
    str
        Markdown report string.
    """
    lines = [
        "# FWA Mismatch Detection — Evaluation Report\n",
        "## Binary Classification Metrics\n",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")

    if threshold_df is not None:
        lines.extend([
            "\n## Threshold Analysis\n",
            "| Threshold | Precision | Recall | F1 | Flag Rate |",
            "|-----------|-----------|--------|-----|-----------|",
        ])
        for _, row in threshold_df.iterrows():
            lines.append(
                f"| {row['threshold']:.2f} | {row['precision']:.3f} | "
                f"{row['recall']:.3f} | {row['f1']:.3f} | {row['flag_rate']:.3f} |"
            )

    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        logger.info("Evaluation report saved to %s", output_path)

    return report
