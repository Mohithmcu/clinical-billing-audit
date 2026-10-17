"""Unit tests for Power BI dashboard data export module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from clinical_billing_audit.dashboard.data_export import export_all


def test_export_all_powerbi_tables(sample_paired_df: pd.DataFrame, temp_dir: Path):
    """Test exporting all structured CSV tables for Power BI."""
    df = sample_paired_df.copy()
    df["is_flagged"] = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    df["is_mismatch"] = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    df["anomaly_score"] = [0.1, 0.9, 0.2, 0.3, 0.85, 0.15, 0.25, 0.8, 0.1, 0.2]
    df["severity"] = ["NORMAL", "HIGH_RISK", "NORMAL", "NORMAL", "HIGH_RISK", "NORMAL", "NORMAL", "MEDIUM_RISK", "NORMAL", "NORMAL"]
    df["billed_category"] = df["code_category"].copy()

    metrics = {
        "precision": 0.85,
        "recall": 0.80,
        "f1_score": 0.824,
        "pr_auc": 0.88,
        "roc_auc": 0.91,
    }

    threshold_df = pd.DataFrame({
        "threshold": [0.3, 0.5, 0.7],
        "precision": [0.6, 0.85, 0.95],
        "recall": [0.95, 0.80, 0.50],
        "f1": [0.73, 0.82, 0.65],
    })

    importance_df = pd.DataFrame({
        "feature": ["f1", "f2"],
        "importance": [0.4, 0.6],
    })

    export_all(
        results_df=df,
        metrics=metrics,
        threshold_df=threshold_df,
        importance_df=importance_df,
        output_dir=temp_dir,
    )

    # Verify expected files exist
    assert (temp_dir / "flag_summary.csv").exists()
    assert (temp_dir / "flag_by_specialty.csv").exists()
    assert (temp_dir / "provider_scores.csv").exists()
    assert (temp_dir / "mismatch_details.csv").exists()
    assert (temp_dir / "model_performance.csv").exists()
    assert (temp_dir / "confusion_matrix.csv").exists()
    assert (temp_dir / "feature_importance.csv").exists()
