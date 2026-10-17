"""Unit tests for the evaluation framework."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from clinical_billing_audit.evaluation.error_analysis import (
    error_breakdown,
    error_by_mismatch_type,
    error_by_note_length,
    top_false_positives,
)
from clinical_billing_audit.evaluation.human_review import (
    compare_with_model,
    create_review_sample,
    generate_labeling_template,
)
from clinical_billing_audit.evaluation.metrics import (
    compute_binary_metrics,
    compute_threshold_sweep,
    generate_evaluation_report,
)


def test_compute_binary_metrics():
    """Test standard binary classification metrics calculation."""
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.4, 0.05, 0.3, 0.85, 0.95])

    metrics = compute_binary_metrics(y_true, y_pred, y_score)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert metrics["precision"] == 1.0  # 3 TP, 0 FP
    assert metrics["recall"] == 0.75  # 3 TP, 1 FN


def test_compute_threshold_sweep():
    """Test precision-recall threshold sweep data generation."""
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.8, 0.2, 0.9])
    sweep_df = compute_threshold_sweep(y_true, y_score)
    assert isinstance(sweep_df, pd.DataFrame)
    assert "precision" in sweep_df.columns
    assert "recall" in sweep_df.columns
    assert "threshold" in sweep_df.columns


def test_generate_evaluation_report(temp_dir: Path):
    """Test generation of markdown evaluation report."""
    metrics = {"precision": 0.85, "recall": 0.80, "f1": 0.824, "roc_auc": 0.91}
    out_md = temp_dir / "report.md"
    generate_evaluation_report(metrics, output_path=out_md)
    assert out_md.exists()
    content = out_md.read_text(encoding="utf-8")
    assert "Evaluation Report" in content
    assert "0.8500" in content


def test_error_analysis_breakdowns(sample_paired_df: pd.DataFrame):
    """Test error analysis breakdowns by specialty, mismatch type, note length, and top FPs."""
    df = sample_paired_df.copy()
    df["is_flagged"] = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    df["anomaly_score"] = [0.1, 0.9, 0.2, 0.3, 0.8, 0.15, 0.25, 0.4, 0.1, 0.2]
    df["is_mismatch"] = df["is_injected_mismatch"]

    by_spec = error_breakdown(
        df, y_true_col="is_mismatch", y_pred_col="is_flagged", group_col="medical_specialty"
    )
    assert isinstance(by_spec, pd.DataFrame)
    assert not by_spec.empty

    by_mtype = error_by_mismatch_type(df, y_pred_col="is_flagged")
    assert isinstance(by_mtype, pd.DataFrame)

    by_length = error_by_note_length(
        df, text_col="transcription", y_true_col="is_mismatch", y_pred_col="is_flagged", n_bins=2
    )
    assert isinstance(by_length, pd.DataFrame)

    fps = top_false_positives(
        df, score_col="anomaly_score", y_true_col="is_mismatch", y_pred_col="is_flagged"
    )
    assert isinstance(fps, pd.DataFrame)


def test_human_review_workflow(sample_paired_df: pd.DataFrame, temp_dir: Path):
    """Test human-review sampling, template creation, and evaluation."""
    df = sample_paired_df.copy()
    df["is_flagged"] = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    df["anomaly_score"] = [0.1, 0.9, 0.2, 0.3, 0.8, 0.15, 0.25, 0.7, 0.1, 0.2]

    sample = create_review_sample(df, n_samples=6, random_state=42)
    assert len(sample) <= len(df)

    template_csv = temp_dir / "template.csv"
    generate_labeling_template(sample, output_path=template_csv)
    assert template_csv.exists()

    # Simulate completed human reviews
    human_df = pd.DataFrame(
        {
            "note_id": sample["note_id"],
            "human_label": [
                "correct_match",
                "upcoding",
                "correct_match",
                "correct_match",
                "miscoding",
                "correct_match",
            ][: len(sample)],
        }
    )
    comparison = compare_with_model(human_df, df, join_col="note_id")
    assert isinstance(comparison, dict)
    assert "agreement_rate" in comparison
