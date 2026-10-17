#!/usr/bin/env python3
"""Export pipeline results for Power BI dashboard.

Standalone script to (re-)generate all Power BI CSVs from existing results.

Usage
-----
    python scripts/export_powerbi_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from clinical_billing_audit.dashboard.data_export import export_all
from clinical_billing_audit.evaluation.metrics import (
    compute_binary_metrics,
    compute_threshold_sweep,
)


def main() -> None:
    results_path = Path("data/processed/flagged_results.csv")
    if not results_path.exists():
        print(f"❌ Results not found at {results_path}. Run the pipeline first.")
        return

    df = pd.read_csv(results_path)

    metrics = {}
    threshold_df = None
    if "is_mismatch" in df.columns and "ensemble_flag" in df.columns:
        y_true = df["is_mismatch"].astype(int).values
        y_pred = df["ensemble_flag"].astype(int).values
        y_score = df.get("ensemble_score", pd.Series(0.5, index=df.index)).values
        metrics = compute_binary_metrics(y_true, y_pred, y_score)
        threshold_df = compute_threshold_sweep(y_true, y_score)

    importance_path = Path("reports/powerbi/feature_importance.csv")
    importance_df = pd.read_csv(importance_path) if importance_path.exists() else None

    export_all(
        df,
        metrics=metrics,
        threshold_df=threshold_df,
        importance_df=importance_df,
        output_dir="reports/powerbi",
    )

    print("✅ Power BI exports complete → reports/powerbi/")


if __name__ == "__main__":
    main()
