"""Human-review simulation and hand-labeled dataset generation.

Creates a stratified sample for manual review, generates a labeling
template, and provides functions to compare human labels against model
predictions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_review_sample(
    df: pd.DataFrame,
    n_samples: int = 120,
    stratify_col: str = "medical_specialty",
    include_predicted_mismatches: float = 0.5,
    score_col: str = "anomaly_score",
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a stratified sample for human review.

    Ensures the sample contains a mix of:
    - Known mismatches (from injection)
    - Predicted mismatches (model flagged)
    - Clean/normal cases

    Parameters
    ----------
    df : pd.DataFrame
        Full results dataset.
    n_samples : int
        Target sample size.
    stratify_col : str
        Column for stratified sampling.
    include_predicted_mismatches : float
        Fraction of sample to draw from model-flagged cases.
    score_col : str
        Anomaly score column for ranking.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Sampled DataFrame for review.
    """
    n_flagged = int(n_samples * include_predicted_mismatches)
    n_normal = n_samples - n_flagged

    # Sample from flagged
    if "is_flagged" in df.columns:
        flagged = df[df["is_flagged"].astype(bool)]
        normal = df[~df["is_flagged"].astype(bool)]
    elif score_col in df.columns:
        median_score = df[score_col].median()
        flagged = df[df[score_col] >= median_score]
        normal = df[df[score_col] < median_score]
    else:
        flagged = df.sample(n=n_flagged, random_state=random_state)
        normal = df.drop(flagged.index)

    # Sample
    n_flagged_actual = min(n_flagged, len(flagged))
    n_normal_actual = min(n_normal, len(normal))

    flagged_sample = flagged.sample(n=n_flagged_actual, random_state=random_state)
    normal_sample = normal.sample(n=n_normal_actual, random_state=random_state)

    sample = (
        pd.concat([flagged_sample, normal_sample])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    logger.info(
        "Created review sample: %d total (%d flagged, %d normal)",
        len(sample),
        n_flagged_actual,
        n_normal_actual,
    )
    return sample


def generate_labeling_template(
    sample_df: pd.DataFrame,
    output_path: str | Path,
    text_col: str = "transcription",
    max_text_preview: int = 500,
) -> None:
    """Generate a CSV labeling template for human review.

    The template includes note excerpts, model predictions, and empty columns
    for human labels and comments.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Review sample.
    output_path : str | Path
        Where to save the CSV.
    text_col : str
        Text column name.
    max_text_preview : int
        Maximum characters of text to include in preview.
    """
    template = pd.DataFrame()

    # Columns for the reviewer
    if "note_id" in sample_df.columns:
        template["note_id"] = sample_df["note_id"]
    if "pair_id" in sample_df.columns:
        template["pair_id"] = sample_df["pair_id"]

    if text_col in sample_df.columns:
        template["note_preview"] = (
            sample_df[text_col].astype(str).str[:max_text_preview].str.replace("\n", " ")
        )

    if "medical_specialty" in sample_df.columns:
        template["specialty"] = sample_df["medical_specialty"]
    if "code_category" in sample_df.columns:
        template["true_category"] = sample_df["code_category"]
    if "billed_category" in sample_df.columns:
        template["billed_category"] = sample_df["billed_category"]
    if "anomaly_score" in sample_df.columns:
        template["model_anomaly_score"] = sample_df["anomaly_score"].round(3)
    if "severity" in sample_df.columns:
        template["model_severity"] = sample_df["severity"]

    # Empty columns for human review
    template["human_label"] = ""  # correct_match / upcoding / miscoding / undercoding
    template["human_confidence"] = ""  # high / medium / low
    template["human_notes"] = ""  # Free text

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    logger.info("Labeling template saved to %s (%d rows)", output_path, len(template))


def load_human_labels(filepath: str | Path) -> pd.DataFrame:
    """Load completed human labels.

    Parameters
    ----------
    filepath : str | Path
        Path to the filled-in labeling template.

    Returns
    -------
    pd.DataFrame
        Human-labeled data.
    """
    df = pd.read_csv(filepath)
    # Filter rows that were actually labeled
    labeled = df[df["human_label"].notna() & (df["human_label"] != "")]
    logger.info("Loaded %d human labels from %s", len(labeled), filepath)
    return labeled


def compare_with_model(
    human_df: pd.DataFrame,
    model_df: pd.DataFrame,
    human_label_col: str = "human_label",
    model_flag_col: str = "is_flagged",
    join_col: str = "note_id",
) -> dict:
    """Compare human labels against model predictions.

    Parameters
    ----------
    human_df : pd.DataFrame
        Human-labeled data.
    model_df : pd.DataFrame
        Model results.
    human_label_col : str
        Human label column name.
    model_flag_col : str
        Model prediction column name.
    join_col : str
        Column to join on.

    Returns
    -------
    dict
        Agreement metrics.
    """
    merged = human_df.merge(model_df, on=join_col, how="inner", suffixes=("_human", "_model"))

    if len(merged) == 0:
        logger.warning("No matching records found between human and model data.")
        return {}

    # Convert human labels to binary
    human_is_mismatch = merged[human_label_col].isin(["upcoding", "miscoding", "undercoding"])

    # Model prediction
    if model_flag_col in merged.columns:
        model_is_flagged = merged[model_flag_col].astype(bool)
    else:
        model_is_flagged = pd.Series(False, index=merged.index)

    agreement = (human_is_mismatch == model_is_flagged).mean()

    from clinical_billing_audit.evaluation.metrics import compute_binary_metrics

    metrics = compute_binary_metrics(
        human_is_mismatch.astype(int),
        model_is_flagged.astype(int),
    )
    metrics["agreement_rate"] = float(agreement)
    metrics["n_reviewed"] = len(merged)

    logger.info(
        "Human vs. Model comparison: %d samples, %.1f%% agreement, P=%.2f R=%.2f",
        len(merged),
        100 * agreement,
        metrics["precision"],
        metrics["recall"],
    )
    return metrics
