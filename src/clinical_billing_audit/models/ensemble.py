"""Ensemble mismatch scorer — combines classifier + anomaly + NER signals.

Fuses the outputs of the baseline classifier (or transformer), the anomaly
detector, and NER-based code predictions into a single mismatch score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)


class EnsembleMismatchScorer:
    """Combine multiple signals into a single mismatch score.

    Score Components (default weights)
    -----------------------------------
    1. Anomaly score (weight=0.35): From Isolation Forest
    2. Classifier mismatch (weight=0.30): Predicted category ≠ billed category
    3. NER mismatch (weight=0.20): NER-derived category ≠ billed category
    4. Confidence penalty (weight=0.15): Low classifier confidence → higher score

    Parameters
    ----------
    weights : dict[str, float] | None
        Component weights. Keys: ``anomaly``, ``classifier``, ``ner``, ``confidence``.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        default_weights = {
            "anomaly": 0.35,
            "classifier": 0.30,
            "ner": 0.20,
            "confidence": 0.15,
        }
        self.weights = {**default_weights, **(weights or {})}

        # Normalise weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def score(
        self,
        df: pd.DataFrame,
        anomaly_scores: np.ndarray | None = None,
        classifier_preds: list[str] | None = None,
        classifier_confidence: np.ndarray | None = None,
        ner_predicted_categories: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute ensemble mismatch scores.

        Parameters
        ----------
        df : pd.DataFrame
            Paired dataset with ``billed_category`` column.
        anomaly_scores : np.ndarray | None
            Anomaly scores from Isolation Forest (0–1).
        classifier_preds : list[str] | None
            Predicted categories from the classifier.
        classifier_confidence : np.ndarray | None
            Max class probability from classifier.
        ner_predicted_categories : list[str] | None
            Predicted categories from NER entity mapping.

        Returns
        -------
        pd.DataFrame
            DataFrame with ensemble scores and component breakdown.
        """
        result = df.copy()

        # Component 1: Anomaly score
        if anomaly_scores is not None:
            result["score_anomaly"] = anomaly_scores
        else:
            result["score_anomaly"] = 0.5

        # Component 2: Classifier mismatch
        if classifier_preds is not None and "billed_category" in df.columns:
            result["score_classifier"] = np.array(
                [
                    0.0 if pred == billed else 1.0
                    for pred, billed in zip(classifier_preds, df["billed_category"])
                ]
            )
        else:
            result["score_classifier"] = 0.0

        # Component 3: NER mismatch
        if ner_predicted_categories is not None and "billed_category" in df.columns:
            result["score_ner"] = np.array(
                [
                    0.0 if ner_cat == billed else 0.8
                    for ner_cat, billed in zip(ner_predicted_categories, df["billed_category"])
                ]
            )
        else:
            result["score_ner"] = 0.0

        # Component 4: Confidence penalty
        if classifier_confidence is not None:
            result["score_confidence"] = 1.0 - classifier_confidence
        else:
            result["score_confidence"] = 0.5

        # Ensemble score
        result["ensemble_score"] = (
            self.weights["anomaly"] * result["score_anomaly"]
            + self.weights["classifier"] * result["score_classifier"]
            + self.weights["ner"] * result["score_ner"]
            + self.weights["confidence"] * result["score_confidence"]
        )

        # Classify
        result["ensemble_flag"] = result["ensemble_score"] >= 0.5
        result["ensemble_severity"] = result["ensemble_score"].apply(
            lambda s: (
                "HIGH_RISK"
                if s >= 0.75
                else ("MEDIUM_RISK" if s >= 0.55 else ("LOW_RISK" if s >= 0.40 else "NORMAL"))
            )
        )

        logger.info(
            "Ensemble scoring: %d flagged (%.1f%%), mean score: %.3f",
            result["ensemble_flag"].sum(),
            100 * result["ensemble_flag"].mean(),
            result["ensemble_score"].mean(),
        )

        return result

    def get_component_summary(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """Summarise the contribution of each component.

        Parameters
        ----------
        scored_df : pd.DataFrame
            Output of :meth:`score`.

        Returns
        -------
        pd.DataFrame
            Component statistics.
        """
        components = ["score_anomaly", "score_classifier", "score_ner", "score_confidence"]
        rows = []
        for comp in components:
            if comp in scored_df.columns:
                rows.append(
                    {
                        "component": comp.replace("score_", ""),
                        "weight": self.weights.get(comp.replace("score_", ""), 0),
                        "mean": float(scored_df[comp].mean()),
                        "std": float(scored_df[comp].std()),
                        "min": float(scored_df[comp].min()),
                        "max": float(scored_df[comp].max()),
                    }
                )
        return pd.DataFrame(rows)
