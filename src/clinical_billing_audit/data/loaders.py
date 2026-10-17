"""Data loaders for MTSamples clinical notes and CMS DE-SynPUF claims."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_mtsamples(filepath: str | Path) -> pd.DataFrame:
    """Load the MTSamples clinical-notes dataset from CSV.

    Expected columns (Kaggle "Medical Transcriptions" format):
      - description, medical_specialty, sample_name, transcription, keywords

    Parameters
    ----------
    filepath : str | Path
        Path to the MTSamples CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with a ``note_id`` column added.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"MTSamples file not found at {filepath}. "
            "Download it from https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions"
        )

    df = pd.read_csv(filepath, encoding="utf-8")

    # Standardise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Add a deterministic note ID
    df.insert(0, "note_id", [f"MT_{i:05d}" for i in range(len(df))])

    # Ensure the text column exists
    text_col = None
    for candidate in ("transcription", "text", "note_text", "clinical_note"):
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col and text_col != "transcription":
        df = df.rename(columns={text_col: "transcription"})
    if "transcription" not in df.columns:
        raise ValueError(f"Cannot find text column in MTSamples. Columns: {list(df.columns)}")

    logger.info("Loaded %d MTSamples notes from %s", len(df), filepath)
    return df


def load_cms_claims(
    filepath: str | Path,
    claim_type: str = "inpatient",
) -> pd.DataFrame:
    """Load CMS DE-SynPUF synthetic claims.

    Parameters
    ----------
    filepath : str | Path
        Path to the CMS claims CSV file.
    claim_type : str
        Type of claims file: ``'inpatient'``, ``'outpatient'``, or ``'carrier'``.

    Returns
    -------
    pd.DataFrame
        Loaded claims DataFrame with standardised column names.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"CMS claims file not found at {filepath}. "
            "Download from https://www.cms.gov/data-research/statistics-trends-and-reports/"
            "medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-"
            "synthetic-public-use-file-de-synpuf"
        )

    df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip().str.upper()

    logger.info(
        "Loaded %d %s claims from %s (columns: %d)",
        len(df),
        claim_type,
        filepath,
        len(df.columns),
    )
    return df


def load_icd9_mapping(filepath: str | Path | None = None) -> dict[str, str]:
    """Load an ICD-9 code → description mapping.

    If no file is provided, returns a built-in subset covering the most
    common diagnosis codes seen in CMS SynPUF data.

    Parameters
    ----------
    filepath : str | Path | None
        Optional CSV with columns ``code`` and ``description``.

    Returns
    -------
    dict[str, str]
        Mapping from ICD-9 code to human-readable description.
    """
    if filepath and Path(filepath).exists():
        df = pd.read_csv(filepath)
        return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))

    # Built-in common codes (CMS SynPUF top-50 approximate)
    return {
        "4019": "Essential hypertension, unspecified",
        "25000": "Diabetes mellitus type II, not stated",
        "2724": "Hyperlipidemia",
        "4011": "Essential hypertension, benign",
        "42731": "Atrial fibrillation",
        "41401": "Coronary atherosclerosis of native vessel",
        "4280": "Congestive heart failure, unspecified",
        "40390": "Hypertensive chronic kidney disease",
        "486": "Pneumonia, organism unspecified",
        "51881": "Acute respiratory failure",
        "49390": "Asthma, unspecified",
        "5990": "Urinary tract infection, unspecified",
        "78079": "Malaise and fatigue",
        "27801": "Morbid obesity",
        "78650": "Chest pain, unspecified",
        "71590": "Osteoarthrosis, unspecified",
        "7140": "Rheumatoid arthritis",
        "5849": "Acute kidney failure, unspecified",
        "311": "Depressive disorder, NEC",
        "V5869": "Long-term use of other medications",
    }
