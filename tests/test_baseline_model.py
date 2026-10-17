"""Unit tests for the Baseline TF-IDF + Logistic Regression classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from clinical_billing_audit.models.baseline import BaselineClassifier


def test_baseline_classifier_init():
    """Test model initialization."""
    clf = BaselineClassifier()
    assert not clf._is_fitted
    assert clf.tfidf_params["max_features"] == 15000


def test_baseline_classifier_fit_predict(sample_paired_df, temp_dir: Path):
    """Test fitting, predicting, probabilities, and serialization."""
    clf = BaselineClassifier(
        tfidf_params={"max_features": 100, "min_df": 1},
        lr_params={"max_iter": 200},
    )

    texts = sample_paired_df["transcription"].tolist()
    labels = sample_paired_df["code_category"].tolist()

    # Fit
    train_metrics = clf.fit(texts, labels)
    assert clf._is_fitted
    assert "train_f1_weighted" in train_metrics
    assert train_metrics["n_classes"] > 1

    # Predict
    preds = clf.predict(texts)
    assert len(preds) == len(texts)
    assert isinstance(preds[0], str)

    # Probabilities
    probs = clf.predict_proba(texts)
    assert probs.shape == (len(texts), train_metrics["n_classes"])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Evaluate
    eval_report = clf.evaluate(texts, labels)
    assert "accuracy" in eval_report
    assert "weighted avg" in eval_report

    # Save & Load
    model_dir = temp_dir / "saved_baseline"
    clf.save(model_dir)
    assert (model_dir / "tfidf_vectorizer.joblib").exists()
    assert (model_dir / "logistic_regression.joblib").exists()
    assert (model_dir / "metadata.json").exists()

    loaded_clf = BaselineClassifier.load(model_dir)
    assert loaded_clf._is_fitted
    loaded_preds = loaded_clf.predict(texts)
    assert np.array_equal(preds, loaded_preds)


def test_baseline_feature_importance(sample_paired_df):
    """Test feature importance retrieval."""
    clf = BaselineClassifier(
        tfidf_params={"max_features": 50, "min_df": 1},
        lr_params={"max_iter": 200},
    )
    texts = sample_paired_df["transcription"].tolist()
    labels = sample_paired_df["code_category"].tolist()
    clf.fit(texts, labels)

    importance_df = clf.get_feature_importance(top_n=5)
    assert not importance_df.empty
    assert "feature" in importance_df.columns
    assert "coefficient" in importance_df.columns
