"""Unit tests for the data quality module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clinical_billing_audit.data.quality import (
    DataQualityChecker,
    QualityReport,
    QualityResult,
    check_class_balance,
    check_duplicates,
    check_length_outliers,
    check_null_fraction,
)


def test_quality_result_dataclass():
    """Test QualityResult properties."""
    res = QualityResult(check_name="test_check", passed=True, message="All good", metric_value=0.0)
    assert res.passed
    assert res.check_name == "test_check"
    assert res.metric_value == 0.0


def test_quality_report_aggregation(temp_dir: Path):
    """Test QualityReport metrics and saving."""
    r1 = QualityResult("c1", True, "ok", metric_value=0.01)
    r2 = QualityResult("c2", False, "fail", metric_value=0.15)
    report = QualityReport(results=[r1, r2])

    assert not report.all_passed
    assert report.n_passed == 1
    assert report.n_failed == 1

    summary = report.summary_df()
    assert len(summary) == 2
    assert "check" in summary.columns

    save_path = temp_dir / "report.csv"
    report.save(save_path)
    assert save_path.exists()


def test_check_null_fraction():
    """Test null fraction detection."""
    df = pd.DataFrame({"a": [1, 2, None, 4], "b": [1, 2, 3, 4]})
    res = check_null_fraction(df, max_fraction=0.10)
    assert not res.passed
    assert res.metric_value == 0.25

    res_pass = check_null_fraction(df, max_fraction=0.30)
    assert res_pass.passed


def test_check_duplicates():
    """Test duplicate check on text column."""
    df = pd.DataFrame({"transcription": ["note 1", "note 1", "note 2", "note 3"]})
    res = check_duplicates(df, subset=["transcription"], max_fraction=0.10)
    assert not res.passed
    assert res.metric_value == 0.25

    res_pass = check_duplicates(df, subset=["transcription"], max_fraction=0.50)
    assert res_pass.passed


def test_check_length_outliers():
    """Test text length outlier check."""
    # 9 normal length notes, 1 extremely short
    df = pd.DataFrame({
        "transcription": ["This is a standard length clinical note transcription." for _ in range(9)] + ["hi"]
    })
    res = check_length_outliers(df, text_col="transcription", min_length=10)
    assert not res.passed
    assert res.flagged_rows is not None
    assert len(res.flagged_rows) >= 1


def test_check_class_balance():
    """Test specialty class balance checking."""
    df = pd.DataFrame({
        "medical_specialty": ["Cardiology"] * 10 + ["Orthopedics"] * 10
    })
    res = check_class_balance(df, label_col="medical_specialty", min_samples_per_class=5)
    assert res.passed

    # Severe imbalance
    df_imbalanced = pd.DataFrame({
        "medical_specialty": ["Cardiology"] * 100 + ["RareSpecialty"] * 1
    })
    res_imb = check_class_balance(df_imbalanced, label_col="medical_specialty", min_samples_per_class=5)
    assert not res_imb.passed


def test_data_quality_checker_orchestrator(sample_notes_df: pd.DataFrame):
    """Test full DataQualityChecker on clinical notes."""
    checker = DataQualityChecker()
    report = checker.run(sample_notes_df, dataset_type="notes")
    assert isinstance(report, QualityReport)
    assert len(report.results) >= 3
