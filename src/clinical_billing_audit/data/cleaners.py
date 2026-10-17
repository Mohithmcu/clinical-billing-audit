"""Data cleaning and profiling for clinical notes and claims."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)


# ── Text Cleaning ───────────────────────────────────────────


def clean_clinical_notes(
    df: pd.DataFrame,
    text_col: str = "transcription",
    min_length: int = 50,
    max_length: int = 50_000,
) -> pd.DataFrame:
    """Clean clinical note text and remove invalid rows.

    Steps
    -----
    1. Drop rows with null text.
    2. Strip whitespace and normalise line breaks.
    3. Remove PHI-like placeholder patterns (e.g., ``[**...**]``).
    4. Expand common clinical abbreviations inline.
    5. Remove notes shorter than *min_length* or longer than *max_length*.
    6. Drop exact-duplicate notes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a text column.
    text_col : str
        Name of the text column.
    min_length : int
        Minimum character count to keep a note.
    max_length : int
        Maximum character count to keep a note.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (copy).
    """
    df = df.copy()
    n_start = len(df)

    # 1. Drop null text
    mask_null = df[text_col].isna()
    df = df[~mask_null]
    n_after_null = len(df)
    logger.info("Dropped %d rows with null text", n_start - n_after_null)

    # 2. Normalise whitespace
    df[text_col] = (
        df[text_col]
        .astype(str)
        .str.replace(r"\r\n", "\n", regex=True)
        .str.replace(r"\r", "\n", regex=True)
        .str.replace(r"[ \t]+", " ", regex=True)
        .str.strip()
    )

    # 3. Remove PHI-like placeholders
    df[text_col] = df[text_col].str.replace(r"\[\*\*[^\]]*\*\*\]", "[REDACTED]", regex=True)

    # 4. Remove very short/long notes
    lengths = df[text_col].str.len()
    mask_length = (lengths >= min_length) & (lengths <= max_length)
    n_before_length = len(df)
    df = df[mask_length]
    logger.info(
        "Dropped %d notes outside length range [%d, %d]",
        n_before_length - len(df),
        min_length,
        max_length,
    )

    # 5. Drop exact duplicates
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=[text_col], keep="first")
    logger.info("Dropped %d exact-duplicate notes", n_before_dedup - len(df))

    df = df.reset_index(drop=True)
    logger.info(
        "Cleaning complete: %d → %d notes (%.1f%% retained)",
        n_start,
        len(df),
        100 * len(df) / max(n_start, 1),
    )
    return df


def clean_specialty_labels(df: pd.DataFrame, col: str = "medical_specialty") -> pd.DataFrame:
    """Standardise specialty labels: strip, title-case, collapse near-duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a specialty column.
    col : str
        Specialty column name.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned specialty labels.
    """
    df = df.copy()
    if col not in df.columns:
        logger.warning("Specialty column '%s' not found — skipping.", col)
        return df

    df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    # Remove leading/trailing whitespace and hyphens
    df[col] = df[col].str.strip(" -")

    # Collapse nan-like strings
    df.loc[df[col].isin(["nan", "NaN", "None", ""]), col] = np.nan

    n_null = df[col].isna().sum()
    if n_null > 0:
        logger.info("Found %d notes with null specialty — will be dropped.", n_null)
        df = df.dropna(subset=[col])

    logger.info("Unique specialties after cleaning: %d", df[col].nunique())
    return df.reset_index(drop=True)


# ── Data Profiling ──────────────────────────────────────────


def profile_dataset(
    df: pd.DataFrame,
    text_col: str = "transcription",
    label_col: str = "medical_specialty",
    output_path: str | Path | None = None,
) -> dict:
    """Generate a statistical profile of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    text_col : str
        Name of the clinical-note text column.
    label_col : str
        Name of the class/specialty label column.
    output_path : str | Path | None
        If provided, writes a profile summary CSV.

    Returns
    -------
    dict
        Profile statistics.
    """
    profile: dict = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
    }

    # Null counts
    null_counts = df.isna().sum()
    profile["null_counts"] = null_counts.to_dict()
    profile["null_fractions"] = (null_counts / len(df)).to_dict()

    # Text length statistics
    if text_col in df.columns:
        lengths = df[text_col].astype(str).str.len()
        profile["text_length"] = {
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "std": float(lengths.std()),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "q1": float(lengths.quantile(0.25)),
            "q3": float(lengths.quantile(0.75)),
        }

    # Label distribution
    if label_col in df.columns:
        label_dist = df[label_col].value_counts()
        profile["label_distribution"] = label_dist.to_dict()
        profile["n_labels"] = int(label_dist.shape[0])
        profile["label_balance_ratio"] = float(label_dist.min() / label_dist.max())

    # Duplicate check
    if text_col in df.columns:
        n_dupes = df.duplicated(subset=[text_col]).sum()
        profile["n_duplicates"] = int(n_dupes)
        profile["duplicate_fraction"] = float(n_dupes / max(len(df), 1))

    # Write summary
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame(
            [
                {"metric": k, "value": str(v)}
                for k, v in profile.items()
                if not isinstance(v, (dict, list))
            ]
        )
        summary_df.to_csv(output_path, index=False)
        logger.info("Profile saved to %s", output_path)

    return profile


def clean_claims_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean CMS SynPUF claims data.

    Steps
    -----
    1. Drop rows with no diagnosis codes at all.
    2. Standardise diagnosis/procedure code columns to strings.
    3. Fill NaN codes with empty string.
    4. Parse date columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw CMS claims DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned claims DataFrame.
    """
    df = df.copy()
    n_start = len(df)

    # Identify diagnosis columns
    diag_cols = [c for c in df.columns if "DGNS" in c]
    proc_cols = [c for c in df.columns if "PRCDR" in c]
    hcpcs_cols = [c for c in df.columns if "HCPCS" in c]

    code_cols = diag_cols + proc_cols + hcpcs_cols

    # Convert code columns to string
    for col in code_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": "", "NaN": "", "None": ""})

    # Drop rows where ALL diagnosis codes are empty
    if diag_cols:
        has_any_diag = df[diag_cols].apply(lambda row: any(v != "" for v in row), axis=1)
        df = df[has_any_diag]
        logger.info("Dropped %d claims with no diagnosis codes", n_start - len(df))

    # Parse dates
    for date_col in [c for c in df.columns if "DT" in c]:
        df[date_col] = pd.to_datetime(df[date_col], format="%Y%m%d", errors="coerce")

    df = df.reset_index(drop=True)
    logger.info("Claims cleaning complete: %d → %d rows", n_start, len(df))
    return df
