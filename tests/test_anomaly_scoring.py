"""Unit tests for Anomaly Detection and Ensemble Mismatch Scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from clinical_billing_audit.models.anomaly import AnomalyDetector
from clinical_billing_audit.models.ensemble import EnsembleMismatchScorer


def test_anomaly_detector_feature_engineering(sample_paired_df: pd.DataFrame):
    """Test feature engineering for anomaly detection."""
    df = sample_paired_df.copy()
    df["billed_category"] = df["code_category"].copy()

    detector = AnomalyDetector(n_estimators=20)
    features = detector.engineer_features(df)

    assert isinstance(features, pd.DataFrame)
    assert len(features) == len(df)
    assert "code_frequency_deviation" in features.columns
    assert "ner_vs_billed_similarity" in features.columns
    assert "note_length_ratio" in features.columns


def test_anomaly_detector_fit_score(sample_paired_df: pd.DataFrame, temp_dir: Path):
    """Test fitting Isolation Forest and computing anomaly scores."""
    df = sample_paired_df.copy()
    df["billed_category"] = df["code_category"].copy()

    detector = AnomalyDetector(n_estimators=30, contamination=0.2)
    features = detector.engineer_features(df)
    fit_metrics = detector.fit(features)

    assert detector._is_fitted
    assert "n_samples" in fit_metrics
    assert fit_metrics["n_samples"] == len(df)

    # Score
    scores = detector.score(features)
    assert len(scores) == len(df)
    assert all(0.0 <= s <= 1.0 for s in scores)

    # Flagging
    flagged_df = detector.flag_mismatches(df, features)
    assert "anomaly_score" in flagged_df.columns
    assert "is_flagged" in flagged_df.columns
    assert "severity" in flagged_df.columns

    # Save and Load
    model_dir = temp_dir / "saved_detector"
    detector.save(model_dir)
    assert (model_dir / "isolation_forest.joblib").exists()
    assert (model_dir / "anomaly_metadata.json").exists()

    loaded_detector = AnomalyDetector.load(model_dir)
    loaded_scores = loaded_detector.score(features)
    np.testing.assert_allclose(scores, loaded_scores, atol=1e-5)


def test_ensemble_scorer(sample_paired_df: pd.DataFrame):
    """Test ensemble mismatch scoring combining rules + anomaly scores."""
    df = sample_paired_df.copy()
    df["billed_category"] = df["code_category"].copy()
    anomaly_scores = np.array([0.1, 0.9, 0.2, 0.3, 0.85, 0.15, 0.25, 0.8, 0.1, 0.2])

    scorer = EnsembleMismatchScorer()
    results_df = scorer.score(df, anomaly_scores=anomaly_scores)
    assert "ensemble_score" in results_df.columns
    assert "ensemble_flag" in results_df.columns
    assert "ensemble_severity" in results_df.columns

    comp_summary = scorer.get_component_summary(results_df)
    assert isinstance(comp_summary, pd.DataFrame)
    assert not comp_summary.empty
