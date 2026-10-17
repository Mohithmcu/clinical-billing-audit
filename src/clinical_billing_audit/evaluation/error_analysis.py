"""Error analysis for mismatch detection.

Breaks down false positives and false negatives by:
- Specialty / code category
- Mismatch type (upcoding, miscoding, undercoding)
- Entity density
- Note length
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def error_breakdown(
    df: pd.DataFrame,
    y_true_col: str = "is_mismatch",
    y_pred_col: str = "is_flagged",
    group_col: str = "medical_specialty",
) -> pd.DataFrame:
    """Break down errors by a grouping column.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame with true/pred columns.
    y_true_col : str
        Ground-truth binary column.
    y_pred_col : str
        Predicted binary column.
    group_col : str
        Column to group by.

    Returns
    -------
    pd.DataFrame
        Error breakdown table.
    """
    if group_col not in df.columns:
        logger.warning("Group column '%s' not found.", group_col)
        return pd.DataFrame()

    rows = []
    for group, gdf in df.groupby(group_col):
        y_true = gdf[y_true_col].astype(int)
        y_pred = gdf[y_pred_col].astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        rows.append({
            "group": group,
            "n_samples": len(gdf),
            "n_true_mismatches": int(y_true.sum()),
            "n_flagged": int(y_pred.sum()),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    result = pd.DataFrame(rows).sort_values("n_samples", ascending=False)
    return result


def error_by_mismatch_type(
    df: pd.DataFrame,
    y_pred_col: str = "is_flagged",
) -> pd.DataFrame:
    """Analyse detection rates by mismatch type (upcoding, miscoding, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        Results with ``mismatch_type`` and prediction columns.
    y_pred_col : str
        Predicted binary column.

    Returns
    -------
    pd.DataFrame
        Detection rate per mismatch type.
    """
    if "mismatch_type" not in df.columns:
        return pd.DataFrame()

    rows = []
    for mtype, mdf in df.groupby("mismatch_type"):
        detected = mdf[y_pred_col].astype(int).sum()
        total = len(mdf)
        rows.append({
            "mismatch_type": mtype,
            "total": total,
            "detected": int(detected),
            "detection_rate": detected / max(total, 1),
            "missed": total - detected,
        })

    return pd.DataFrame(rows).sort_values("detection_rate", ascending=False)


def error_by_note_length(
    df: pd.DataFrame,
    text_col: str = "transcription",
    y_true_col: str = "is_mismatch",
    y_pred_col: str = "is_flagged",
    n_bins: int = 5,
) -> pd.DataFrame:
    """Analyse error rates by note length quartiles.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    text_col : str
        Text column.
    y_true_col, y_pred_col : str
        Ground-truth and prediction columns.
    n_bins : int
        Number of length bins.

    Returns
    -------
    pd.DataFrame
        Error rates per length bin.
    """
    if text_col not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["_note_length"] = df[text_col].astype(str).str.len()
    df["_length_bin"] = pd.qcut(df["_note_length"], q=n_bins, duplicates="drop")

    rows = []
    for bin_label, bdf in df.groupby("_length_bin", observed=True):
        y_true = bdf[y_true_col].astype(int)
        y_pred = bdf[y_pred_col].astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        rows.append({
            "length_bin": str(bin_label),
            "n_samples": len(bdf),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "fp_rate": fp / max(len(bdf), 1),
        })

    return pd.DataFrame(rows)


def top_false_positives(
    df: pd.DataFrame,
    score_col: str = "anomaly_score",
    y_true_col: str = "is_mismatch",
    y_pred_col: str = "is_flagged",
    n: int = 20,
) -> pd.DataFrame:
    """Return top-N false positives (highest scored correct matches flagged as mismatches).

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    score_col : str
        Anomaly score column.
    y_true_col, y_pred_col : str
        Ground-truth and prediction columns.
    n : int
        Number of top FPs to return.

    Returns
    -------
    pd.DataFrame
        Top false positive cases.
    """
    fps = df[(df[y_true_col] == False) & (df[y_pred_col] == True)]
    if fps.empty:
        return pd.DataFrame()

    cols = [c for c in ["note_id", "medical_specialty", "code_category",
                         "billed_category", score_col, "severity"] if c in fps.columns]
    return fps.nlargest(n, score_col)[cols]


def top_false_negatives(
    df: pd.DataFrame,
    score_col: str = "anomaly_score",
    y_true_col: str = "is_mismatch",
    y_pred_col: str = "is_flagged",
    n: int = 20,
) -> pd.DataFrame:
    """Return top-N false negatives (lowest scored mismatches missed by the model).

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    score_col : str
        Anomaly score column.
    y_true_col, y_pred_col : str
        Ground-truth and prediction columns.
    n : int
        Number of top FNs to return.

    Returns
    -------
    pd.DataFrame
        Top false negative cases.
    """
    fns = df[(df[y_true_col] == True) & (df[y_pred_col] == False)]
    if fns.empty:
        return pd.DataFrame()

    cols = [c for c in ["note_id", "medical_specialty", "code_category",
                         "billed_category", "mismatch_type", score_col,
                         "severity"] if c in fns.columns]
    return fns.nsmallest(n, score_col)[cols]


def plot_error_analysis(
    breakdown_df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> None:
    """Plot error breakdown by group.

    Parameters
    ----------
    breakdown_df : pd.DataFrame
        Output of :func:`error_breakdown`.
    output_path : str | Path | None
        If provided, saves the figure.
    """
    if not HAS_PLOT or breakdown_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Precision/Recall by group
    top_groups = breakdown_df.head(15)
    x = range(len(top_groups))
    axes[0].barh(x, top_groups["precision"], alpha=0.7, label="Precision", color="#2563eb")
    axes[0].barh(x, top_groups["recall"], alpha=0.5, label="Recall", color="#dc2626")
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(top_groups["group"], fontsize=9)
    axes[0].set_xlabel("Score")
    axes[0].set_title("Precision / Recall by Specialty")
    axes[0].legend()

    # FP/FN counts by group
    axes[1].barh(x, top_groups["FP"], alpha=0.7, label="False Positives", color="#f59e0b")
    axes[1].barh(x, -top_groups["FN"], alpha=0.7, label="False Negatives", color="#ef4444")
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(top_groups["group"], fontsize=9)
    axes[1].set_xlabel("Count (FN shown as negative)")
    axes[1].set_title("Error Counts by Specialty")
    axes[1].legend()

    plt.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Error analysis plot saved to %s", output_path)
    plt.close(fig)
