"""Baseline classifier: TF-IDF + Logistic Regression.

A lightweight, CPU-friendly model that classifies clinical notes into
code categories using TF-IDF text features augmented with NER entity
counts from the extraction step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)


class BaselineClassifier:
    """TF-IDF + Logistic Regression classifier for clinical notes.

    Features
    --------
    1. TF-IDF on note text (unigrams + bigrams)
    2. NER entity counts (diagnoses, medications, procedures, anatomy)
    3. Text-length features (character count, word count)

    Parameters
    ----------
    tfidf_params : dict | None
        Parameters for ``TfidfVectorizer``.
    lr_params : dict | None
        Parameters for ``LogisticRegression``.
    """

    def __init__(
        self,
        tfidf_params: dict | None = None,
        lr_params: dict | None = None,
    ) -> None:
        tfidf_defaults = {
            "max_features": 15000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
        }
        lr_defaults = {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "multi_class": "multinomial",
            "class_weight": "balanced",
        }

        self.tfidf_params = {**tfidf_defaults, **(tfidf_params or {})}
        self.lr_params = {**lr_defaults, **(lr_params or {})}

        # Handle ngram_range — may come as list from YAML
        ngram = self.tfidf_params.get("ngram_range", (1, 2))
        if isinstance(ngram, list):
            self.tfidf_params["ngram_range"] = tuple(ngram)

        self.vectorizer = TfidfVectorizer(**self.tfidf_params)
        self.classifier = LogisticRegression(**self.lr_params)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler(with_mean=False)  # Sparse-compatible
        self._is_fitted = False
        self._feature_names: list[str] = []

    def _build_features(
        self,
        texts: list[str],
        ner_features: pd.DataFrame | None = None,
        fit_tfidf: bool = False,
    ) -> Any:
        """Build combined feature matrix.

        Parameters
        ----------
        texts : list[str]
            Raw note texts.
        ner_features : pd.DataFrame | None
            DataFrame with NER entity count columns.
        fit_tfidf : bool
            If True, fit the vectorizer; otherwise transform only.
        """
        # TF-IDF features
        if fit_tfidf:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.vectorizer.transform(texts)

        # Text-length features
        char_lengths = np.array([len(t) for t in texts]).reshape(-1, 1)
        word_lengths = np.array([len(t.split()) for t in texts]).reshape(-1, 1)
        length_features = np.hstack([char_lengths, word_lengths])

        # NER features
        if ner_features is not None and not ner_features.empty:
            ner_cols = [c for c in ner_features.columns if c.startswith("n_")]
            if ner_cols:
                ner_matrix = ner_features[ner_cols].fillna(0).values
                combined = hstack(
                    [
                        tfidf_matrix,
                        length_features,
                        ner_matrix,
                    ]
                )
                self._feature_names = (
                    list(self.vectorizer.get_feature_names_out())
                    + ["char_length", "word_length"]
                    + ner_cols
                )
                return combined

        combined = hstack([tfidf_matrix, length_features])
        self._feature_names = list(self.vectorizer.get_feature_names_out()) + [
            "char_length",
            "word_length",
        ]
        return combined

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        ner_features: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Fit the baseline classifier.

        Parameters
        ----------
        texts : list[str]
            Training note texts.
        labels : list[str]
            Code category labels.
        ner_features : pd.DataFrame | None
            NER entity count features.

        Returns
        -------
        dict
            Training metrics.
        """
        logger.info("Building features for %d training samples...", len(texts))
        X = self._build_features(texts, ner_features, fit_tfidf=True)
        y = self.label_encoder.fit_transform(labels)

        logger.info(
            "Feature matrix: %s, classes: %d",
            X.shape,
            len(self.label_encoder.classes_),
        )

        # Scale
        X = self.scaler.fit_transform(X)

        # Fit
        self.classifier.fit(X, y)
        self._is_fitted = True

        # Training accuracy
        train_preds = self.classifier.predict(X)
        train_f1 = f1_score(y, train_preds, average="weighted")
        logger.info("Training weighted F1: %.4f", train_f1)

        return {
            "train_f1_weighted": float(train_f1),
            "n_classes": len(self.label_encoder.classes_),
            "n_features": X.shape[1],
            "classes": list(self.label_encoder.classes_),
        }

    def predict(
        self,
        texts: list[str],
        ner_features: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Predict code categories for new notes.

        Parameters
        ----------
        texts : list[str]
            Note texts.
        ner_features : pd.DataFrame | None
            NER features.

        Returns
        -------
        np.ndarray
            Predicted category labels (strings).
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        X = self._build_features(texts, ner_features, fit_tfidf=False)
        X = self.scaler.transform(X)
        y_pred = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(
        self,
        texts: list[str],
        ner_features: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Predict class probabilities.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_classes)`` probability matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        X = self._build_features(texts, ner_features, fit_tfidf=False)
        X = self.scaler.transform(X)
        return self.classifier.predict_proba(X)

    def cross_validate(
        self,
        texts: list[str],
        labels: list[str],
        ner_features: pd.DataFrame | None = None,
        cv: int = 5,
    ) -> dict[str, Any]:
        """Run stratified cross-validation.

        Returns
        -------
        dict
            CV metrics (mean and std of F1).
        """
        X = self._build_features(texts, ner_features, fit_tfidf=True)
        y = self.label_encoder.fit_transform(labels)
        X = self.scaler.fit_transform(X)

        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(
            self.classifier,
            X,
            y,
            cv=cv_splitter,
            scoring="f1_weighted",
        )

        logger.info(
            "%d-fold CV weighted F1: %.4f ± %.4f",
            cv,
            scores.mean(),
            scores.std(),
        )
        return {
            "cv_f1_mean": float(scores.mean()),
            "cv_f1_std": float(scores.std()),
            "cv_scores": scores.tolist(),
            "n_folds": cv,
        }

    def evaluate(
        self,
        texts: list[str],
        labels: list[str],
        ner_features: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Evaluate on a held-out test set.

        Returns
        -------
        dict
            Classification report as a dictionary.
        """
        preds = self.predict(texts, ner_features)
        report = classification_report(labels, preds, output_dict=True, zero_division=0)
        logger.info("\n%s", classification_report(labels, preds, zero_division=0))
        return report

    def get_feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        """Get top features by absolute coefficient magnitude.

        Parameters
        ----------
        top_n : int
            Number of top features per class.

        Returns
        -------
        pd.DataFrame
            Feature importance table.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted.")

        coefs = self.classifier.coef_
        classes = self.label_encoder.classes_

        rows = []
        for i, cls in enumerate(classes):
            if i >= coefs.shape[0]:
                break
            importances = np.abs(coefs[i])
            if issparse(importances):
                importances = importances.toarray().flatten()
            top_indices = np.argsort(importances)[-top_n:][::-1]

            for rank, idx in enumerate(top_indices):
                if idx < len(self._feature_names):
                    rows.append(
                        {
                            "class": cls,
                            "rank": rank + 1,
                            "feature": self._feature_names[idx],
                            "coefficient": float(coefs[i, idx])
                            if not issparse(coefs)
                            else float(coefs[i, idx]),
                            "abs_coefficient": float(importances[idx]),
                        }
                    )

        return pd.DataFrame(rows)

    def save(self, directory: str | Path) -> None:
        """Save model artifacts to disk.

        Parameters
        ----------
        directory : str | Path
            Output directory.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.vectorizer, directory / "tfidf_vectorizer.joblib")
        joblib.dump(self.classifier, directory / "logistic_regression.joblib")
        joblib.dump(self.label_encoder, directory / "label_encoder.joblib")
        joblib.dump(self.scaler, directory / "scaler.joblib")

        meta = {
            "tfidf_params": {
                k: list(v) if isinstance(v, tuple) else v for k, v in self.tfidf_params.items()
            },
            "lr_params": self.lr_params,
            "classes": list(self.label_encoder.classes_),
            "n_features": len(self._feature_names),
        }
        with open(directory / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Baseline model saved to %s", directory)

    @classmethod
    def load(cls, directory: str | Path) -> BaselineClassifier:
        """Load a saved model.

        Parameters
        ----------
        directory : str | Path
            Directory containing saved artifacts.

        Returns
        -------
        BaselineClassifier
            Loaded classifier.
        """
        directory = Path(directory)

        model = cls()
        model.vectorizer = joblib.load(directory / "tfidf_vectorizer.joblib")
        model.classifier = joblib.load(directory / "logistic_regression.joblib")
        model.label_encoder = joblib.load(directory / "label_encoder.joblib")
        model.scaler = joblib.load(directory / "scaler.joblib")
        model._is_fitted = True
        model._feature_names = list(model.vectorizer.get_feature_names_out()) + [
            "char_length",
            "word_length",
        ]

        logger.info("Baseline model loaded from %s", directory)
        return model
