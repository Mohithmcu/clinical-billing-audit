"""Export pipeline results as structured CSVs for Power BI dashboards.

Generates the following tables:
1. flag_summary.csv — Overall flag counts and rates
2. flag_by_specialty.csv — Flags broken down by medical specialty
3. model_performance.csv — Precision, recall, F1 at various thresholds
4. provider_scores.csv — Per-provider anomaly score summaries
5. mismatch_details.csv — Individual flagged case details
6. precision_recall_curve.csv — PR curve data points
7. confusion_matrix.csv — Confusion matrix values
8. feature_importance.csv — Top model features
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)


def export_flag_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """Export overall flag summary."""
    if "is_flagged" not in df.columns:
        logger.warning("No 'is_flagged' column — skipping flag_summary export.")
        return

    summary = {
        "total_pairs": len(df),
        "total_flagged": int(df["is_flagged"].sum()),
        "flag_rate": float(df["is_flagged"].mean()),
        "high_risk": int((df.get("severity", pd.Series()) == "HIGH_RISK").sum()),
        "medium_risk": int((df.get("severity", pd.Series()) == "MEDIUM_RISK").sum()),
        "low_risk": int((df.get("severity", pd.Series()) == "LOW_RISK").sum()),
        "normal": int((df.get("severity", pd.Series()) == "NORMAL").sum()),
    }

    if "is_mismatch" in df.columns:
        true_mismatches = df["is_mismatch"].sum()
        detected = ((df["is_mismatch"]) & (df["is_flagged"])).sum()
        summary["true_mismatches"] = int(true_mismatches)
        summary["true_mismatches_detected"] = int(detected)
        summary["true_detection_rate"] = float(detected / max(true_mismatches, 1))

    pd.DataFrame([summary]).to_csv(output_dir / "flag_summary.csv", index=False)
    logger.info("Exported flag_summary.csv")


def export_flag_by_specialty(df: pd.DataFrame, output_dir: Path) -> None:
    """Export flag counts by medical specialty."""
    if "medical_specialty" not in df.columns or "is_flagged" not in df.columns:
        return

    group = (
        df.groupby("medical_specialty")
        .agg(
            total_pairs=("is_flagged", "count"),
            total_flagged=("is_flagged", "sum"),
            flag_rate=("is_flagged", "mean"),
            avg_anomaly_score=("anomaly_score", "mean")
            if "anomaly_score" in df.columns
            else ("is_flagged", "mean"),
        )
        .reset_index()
        .sort_values("flag_rate", ascending=False)
    )

    if "is_mismatch" in df.columns:
        mismatch_rates = df.groupby("medical_specialty")["is_mismatch"].mean().reset_index()
        mismatch_rates.columns = ["medical_specialty", "true_mismatch_rate"]
        group = group.merge(mismatch_rates, on="medical_specialty", how="left")

    if "severity" in df.columns:
        for sev in ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]:
            group[f"n_{sev.lower()}"] = (
                df[df["severity"] == sev]
                .groupby("medical_specialty")
                .size()
                .reindex(group["medical_specialty"])
                .fillna(0)
                .astype(int)
                .values
            )

    group.to_csv(output_dir / "flag_by_specialty.csv", index=False)
    logger.info("Exported flag_by_specialty.csv")


def export_model_performance(
    metrics: dict,
    threshold_df: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    """Export model performance metrics."""
    # Flat metrics
    flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
    pd.DataFrame([flat]).to_csv(output_dir / "model_performance.csv", index=False)

    # Threshold sweep
    if threshold_df is not None:
        threshold_df.to_csv(output_dir / "precision_recall_curve.csv", index=False)
        logger.info("Exported precision_recall_curve.csv")

    logger.info("Exported model_performance.csv")


def export_provider_scores(df: pd.DataFrame, output_dir: Path) -> None:
    """Export per-provider anomaly score summaries."""
    if "provider_id" not in df.columns:
        return

    agg_dict = {
        "anomaly_score": ["count", "mean", "std", "max"],
    }
    if "is_flagged" in df.columns:
        agg_dict["is_flagged"] = ["sum", "mean"]

    provider = df.groupby("provider_id").agg(agg_dict)
    provider.columns = ["_".join(col).strip("_") for col in provider.columns]
    provider = provider.reset_index().sort_values("anomaly_score_mean", ascending=False)

    provider.to_csv(output_dir / "provider_scores.csv", index=False)
    logger.info("Exported provider_scores.csv")


def export_mismatch_details(
    df: pd.DataFrame,
    output_dir: Path,
    max_rows: int = 500,
    text_preview_length: int = 300,
) -> None:
    """Export individual flagged case details."""
    flagged = df[df.get("is_flagged", pd.Series(dtype=bool)).astype(bool)]
    if flagged.empty:
        logger.warning("No flagged cases to export.")
        return

    # Sort by anomaly score
    if "anomaly_score" in flagged.columns:
        flagged = flagged.sort_values("anomaly_score", ascending=False)

    flagged = flagged.head(max_rows).copy()

    # Truncate text for Power BI
    if "transcription" in flagged.columns:
        flagged["note_preview"] = flagged["transcription"].astype(str).str[:text_preview_length]
        flagged = flagged.drop(columns=["transcription"], errors="ignore")

    export_cols = [
        c
        for c in [
            "pair_id",
            "note_id",
            "medical_specialty",
            "code_category",
            "billed_category",
            "mismatch_type",
            "anomaly_score",
            "severity",
            "severity_label",
            "provider_id",
            "note_preview",
            "ensemble_score",
            "ensemble_severity",
        ]
        if c in flagged.columns
    ]

    flagged[export_cols].to_csv(output_dir / "mismatch_details.csv", index=False)
    logger.info("Exported mismatch_details.csv (%d rows)", len(flagged))


def export_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    output_dir: Path | None = None,
) -> None:
    """Export confusion matrix as CSV."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if labels is None:
        labels = [f"class_{i}" for i in range(cm.shape[0])]

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "true_label"
    cm_df.columns.name = "predicted_label"

    if output_dir:
        cm_df.to_csv(output_dir / "confusion_matrix.csv")
        logger.info("Exported confusion_matrix.csv")


def export_feature_importance(
    importance_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Export feature importance table."""
    if importance_df.empty:
        return
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    logger.info("Exported feature_importance.csv")


def export_all(
    results_df: pd.DataFrame,
    metrics: dict | None = None,
    threshold_df: pd.DataFrame | None = None,
    importance_df: pd.DataFrame | None = None,
    output_dir: str | Path = "reports/powerbi",
) -> None:
    """Run all exports.

    Parameters
    ----------
    results_df : pd.DataFrame
        Full results with flags, scores, severity.
    metrics : dict | None
        Evaluation metrics.
    threshold_df : pd.DataFrame | None
        Threshold sweep data.
    importance_df : pd.DataFrame | None
        Feature importance data.
    output_dir : str | Path
        Export directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_flag_summary(results_df, output_dir)
    export_flag_by_specialty(results_df, output_dir)
    export_provider_scores(results_df, output_dir)
    export_mismatch_details(results_df, output_dir)

    if metrics:
        export_model_performance(metrics, threshold_df, output_dir)

    if "is_mismatch" in results_df.columns and "is_flagged" in results_df.columns:
        export_confusion_matrix(
            results_df["is_mismatch"].astype(int).values,
            results_df["is_flagged"].astype(int).values,
            labels=["Correct", "Mismatch"],
            output_dir=output_dir,
        )

    if importance_df is not None:
        export_feature_importance(importance_df, output_dir)

    logger.info("All Power BI exports complete -> %s", output_dir)
