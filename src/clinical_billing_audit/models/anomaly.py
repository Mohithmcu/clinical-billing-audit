"""Anomaly detection for billing-code mismatch flagging.

Uses Isolation Forest on engineered features to score each note-claim pair
for suspiciousness.  High anomaly scores suggest the billed code doesn't
match what the clinical note supports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from clinical_billing_audit.utils.constants import SEVERITY_LEVELS
from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """Isolation Forest anomaly detector for billing mismatch scoring.

    Feature Engineering
    -------------------
    1. ``code_frequency_deviation``: How rare is the billed code for this specialty?
    2. ``ner_vs_billed_similarity``: Does the NER-predicted category match the billed one?
    3. ``classifier_confidence_gap``: Difference between top-1 and top-2 class probabilities.
    4. ``note_length_code_complexity_ratio``: Note length relative to code complexity.
    5. ``provider_code_distribution_deviation``: How unusual is this provider's coding pattern?

    Parameters
    ----------
    n_estimators : int
        Number of trees in the Isolation Forest.
    contamination : float
        Expected fraction of anomalies.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.15,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

        self.iso_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples="auto",
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._is_fitted = False
        self._feature_names: list[str] = []

    def engineer_features(
        self,
        df: pd.DataFrame,
        ner_df: pd.DataFrame | None = None,
        classifier_probs: np.ndarray | None = None,
        classifier_labels: list[str] | None = None,
    ) -> pd.DataFrame:
        """Build anomaly detection features from the paired dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Paired note-claim dataset with ``code_category``, ``billed_category``,
            ``provider_id``, and ``transcription`` columns.
        ner_df : pd.DataFrame | None
            NER extraction results (entity counts per note).
        classifier_probs : np.ndarray | None
            Classifier probability output (shape: n_samples × n_classes).
        classifier_labels : list[str] | None
            Ordered class labels corresponding to classifier columns.

        Returns
        -------
        pd.DataFrame
            Feature matrix with one row per pair.
        """
        features = pd.DataFrame(index=df.index)

        # Feature 1: Code frequency deviation
        # How rare is the billed code within this specialty?
        if "billed_category" in df.columns and "medical_specialty" in df.columns:
            spec_code_counts = (
                df.groupby(["medical_specialty", "billed_category"])
                .size()
                .reset_index(name="count")
            )
            spec_totals = df.groupby("medical_specialty").size().reset_index(name="total")
            spec_code_counts = spec_code_counts.merge(spec_totals, on="medical_specialty")
            spec_code_counts["frequency"] = spec_code_counts["count"] / spec_code_counts["total"]
            freq_map = dict(
                zip(
                    zip(spec_code_counts["medical_specialty"], spec_code_counts["billed_category"]),
                    spec_code_counts["frequency"],
                )
            )

            features["code_frequency_deviation"] = df.apply(
                lambda row: (
                    1
                    - freq_map.get((row.get("medical_specialty"), row.get("billed_category")), 0.5)
                ),
                axis=1,
            )
        else:
            features["code_frequency_deviation"] = 0.5

        # Feature 2: NER-predicted vs. billed category similarity
        if "code_category" in df.columns and "billed_category" in df.columns:
            features["ner_vs_billed_similarity"] = (
                df["code_category"] != df["billed_category"]
            ).astype(float)
        else:
            features["ner_vs_billed_similarity"] = 0.0

        # Feature 3: Classifier confidence gap
        if classifier_probs is not None and classifier_probs.shape[0] == len(df):
            sorted_probs = np.sort(classifier_probs, axis=1)[:, ::-1]
            if sorted_probs.shape[1] >= 2:
                features["classifier_confidence_gap"] = sorted_probs[:, 0] - sorted_probs[:, 1]
            else:
                features["classifier_confidence_gap"] = sorted_probs[:, 0]

            features["classifier_max_confidence"] = sorted_probs[:, 0]

            # Feature: classifier prediction matches billed category
            if classifier_labels and "billed_category" in df.columns:
                predicted_indices = np.argmax(classifier_probs, axis=1)
                predicted_cats = [classifier_labels[i] for i in predicted_indices]
                features["classifier_billed_match"] = (
                    np.array(predicted_cats) != df["billed_category"].values
                ).astype(float)
            else:
                features["classifier_billed_match"] = 0.0
        else:
            features["classifier_confidence_gap"] = 0.5
            features["classifier_max_confidence"] = 0.5
            features["classifier_billed_match"] = 0.0

        # Feature 4: Note length vs. code complexity ratio
        if "transcription" in df.columns:
            note_lengths = df["transcription"].astype(str).str.len()
            median_length = note_lengths.median()
            features["note_length_ratio"] = note_lengths / max(median_length, 1)
        else:
            features["note_length_ratio"] = 1.0

        # Feature 5: Provider-level code distribution deviation
        if "provider_id" in df.columns and "billed_category" in df.columns:
            prov_counts = pd.crosstab(df["provider_id"], df["billed_category"], normalize="index")
            probs = prov_counts.values
            log_probs = np.log(probs + 1e-10)
            entropies = -np.sum(np.where(probs > 0, probs * log_probs, 0.0), axis=1)
            provider_entropy = dict(zip(prov_counts.index, entropies))
            median_entropy = float(np.median(entropies)) if len(entropies) > 0 else 1.0
            features["provider_entropy_deviation"] = (
                df["provider_id"]
                .map(lambda p: abs(provider_entropy.get(p, median_entropy) - median_entropy))
                .fillna(0.0)
            )
        else:
            features["provider_entropy_deviation"] = 0.0

        # Feature 6: NER entity density (if available)
        if ner_df is not None and "n_entities" in ner_df.columns:
            features["entity_density"] = ner_df["n_entities"].fillna(0).values[: len(df)]
            features["diagnosis_count"] = (
                ner_df.get("n_diagnoses", pd.Series(0)).fillna(0).values[: len(df)]
            )
        else:
            features["entity_density"] = 0.0
            features["diagnosis_count"] = 0.0

        self._feature_names = list(features.columns)
        return features

    def fit(self, feature_df: pd.DataFrame) -> dict[str, Any]:
        """Fit the Isolation Forest on feature data.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Feature matrix.

        Returns
        -------
        dict
            Fitting summary.
        """
        X = feature_df.fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        self.iso_forest.fit(X_scaled)
        self._is_fitted = True

        # Get anomaly scores for training data
        scores = self.iso_forest.decision_function(X_scaled)
        predictions = self.iso_forest.predict(X_scaled)
        n_anomalies = (predictions == -1).sum()

        logger.info(
            "Anomaly detector fitted on %d samples with %d features",
            len(feature_df),
            feature_df.shape[1],
        )
        logger.info(
            "Detected %d anomalies (%.1f%%)",
            n_anomalies,
            100 * n_anomalies / len(feature_df),
        )

        return {
            "n_samples": len(feature_df),
            "n_features": feature_df.shape[1],
            "n_anomalies_detected": int(n_anomalies),
            "anomaly_fraction": float(n_anomalies / len(feature_df)),
            "score_range": (float(scores.min()), float(scores.max())),
        }

    def score(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Score samples for anomaly.

        Scores are normalised to [0, 1] where higher = more anomalous.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Anomaly scores in [0, 1].
        """
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        X = feature_df.fillna(0).values
        X_scaled = self.scaler.transform(X)

        # decision_function: lower = more anomalous
        raw_scores = self.iso_forest.decision_function(X_scaled)

        # Normalise to [0, 1] where 1 = most anomalous
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s > 0:
            normalized = 1 - (raw_scores - min_s) / (max_s - min_s)
        else:
            normalized = np.full_like(raw_scores, 0.5)

        return normalized

    def classify_severity(
        self,
        scores: np.ndarray,
        high_threshold: float = 0.85,
        medium_threshold: float = 0.65,
        low_threshold: float = 0.45,
    ) -> list[str]:
        """Classify anomaly scores into severity levels.

        Parameters
        ----------
        scores : np.ndarray
            Anomaly scores in [0, 1].
        high_threshold, medium_threshold, low_threshold : float
            Score thresholds for each severity level.

        Returns
        -------
        list[str]
            Severity labels.
        """
        severities = []
        for s in scores:
            if s >= high_threshold:
                severities.append("HIGH_RISK")
            elif s >= medium_threshold:
                severities.append("MEDIUM_RISK")
            elif s >= low_threshold:
                severities.append("LOW_RISK")
            else:
                severities.append("NORMAL")
        return severities

    def flag_mismatches(
        self,
        df: pd.DataFrame,
        feature_df: pd.DataFrame,
        high_threshold: float = 0.85,
        medium_threshold: float = 0.65,
        low_threshold: float = 0.45,
    ) -> pd.DataFrame:
        """Score and flag potential billing mismatches.

        Parameters
        ----------
        df : pd.DataFrame
            Original paired dataset.
        feature_df : pd.DataFrame
            Engineered features.
        high_threshold, medium_threshold, low_threshold : float
            Severity thresholds.

        Returns
        -------
        pd.DataFrame
            Original data with anomaly score, severity, and flag columns.
        """
        scores = self.score(feature_df)
        severities = self.classify_severity(scores, high_threshold, medium_threshold, low_threshold)

        result = df.copy()
        result["anomaly_score"] = scores
        result["severity"] = severities
        result["severity_label"] = [SEVERITY_LEVELS.get(s, "Unknown") for s in severities]
        result["is_flagged"] = result["severity"] != "NORMAL"

        # Summary
        flag_counts = result["severity"].value_counts()
        logger.info("Mismatch flagging results:")
        for sev, count in flag_counts.items():
            logger.info("  %s: %d (%.1f%%)", sev, count, 100 * count / len(result))

        return result

    def save(self, directory: str | Path) -> None:
        """Save the anomaly detector to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.iso_forest, directory / "isolation_forest.joblib")
        joblib.dump(self.scaler, directory / "anomaly_scaler.joblib")

        meta = {
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "feature_names": self._feature_names,
        }
        with open(directory / "anomaly_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Anomaly detector saved to %s", directory)

    @classmethod
    def load(cls, directory: str | Path) -> AnomalyDetector:
        """Load a saved anomaly detector."""
        directory = Path(directory)

        with open(directory / "anomaly_metadata.json") as f:
            meta = json.load(f)

        detector = cls(
            n_estimators=meta.get("n_estimators", 200),
            contamination=meta.get("contamination", 0.15),
        )
        detector.iso_forest = joblib.load(directory / "isolation_forest.joblib")
        detector.scaler = joblib.load(directory / "anomaly_scaler.joblib")
        detector._feature_names = meta.get("feature_names", [])
        detector._is_fitted = True

        logger.info("Anomaly detector loaded from %s", directory)
        return detector
