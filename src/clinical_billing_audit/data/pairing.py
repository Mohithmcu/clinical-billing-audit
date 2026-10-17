"""Pair MTSamples notes with CMS SynPUF claims by specialty / code category.

Strategy
--------
1. Map each MTSamples note's *medical_specialty* to a broad code category
   using ``SPECIALTY_TO_CODE_CATEGORY``.
2. For each CMS claim, derive the broad category from the primary ICD-9
   diagnosis code using ``ICD9_BROAD_CATEGORY``.
3. Pair notes and claims within the same category.  Where one side has more
   records, we sample with replacement from the smaller side.
4. Assign synthetic provider IDs to simulate a multi-provider environment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clinical_billing_audit.utils.constants import (
    ICD9_BROAD_CATEGORY,
    SPECIALTY_TO_CODE_CATEGORY,
)
from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)


def _map_specialty_to_category(specialty: str) -> str:
    """Map an MTSamples specialty to a broad code category."""
    return SPECIALTY_TO_CODE_CATEGORY.get(specialty, "general")


def _map_icd9_to_category(code: str) -> str:
    """Map an ICD-9 code to a broad category using the first character/digit."""
    code = str(code).strip()
    if not code or code in ("", "nan"):
        return "unknown"
    first_char = code[0].upper()
    if first_char in ("E", "V"):
        return ICD9_BROAD_CATEGORY.get(first_char, "unknown")
    # Numeric codes: map by first digit range
    try:
        first_digit = int(first_char)
        # ICD-9 numeric ranges (simplified)
        if first_digit == 0:
            return "infectious"
        elif first_digit in (1,):
            return "neoplasms"
        elif first_digit == 2:
            return "endocrine"
        elif first_digit == 3:
            return "nervous"
        elif first_digit == 4:
            return "circulatory"
        elif first_digit == 5:
            return "digestive"
        elif first_digit == 6:
            return "genitourinary"
        elif first_digit == 7:
            return "musculoskeletal"
        elif first_digit in (8, 9):
            return "injury_external"
    except ValueError:
        pass
    return "unknown"


def pair_notes_and_claims(
    notes_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    specialty_col: str = "medical_specialty",
    random_state: int = 42,
    n_providers: int = 50,
) -> pd.DataFrame:
    """Pair clinical notes with claims records by specialty / code category.

    Parameters
    ----------
    notes_df : pd.DataFrame
        Cleaned MTSamples notes.
    claims_df : pd.DataFrame
        Cleaned CMS SynPUF claims.
    specialty_col : str
        Column in *notes_df* with the specialty label.
    random_state : int
        Random seed for reproducibility.
    n_providers : int
        Number of synthetic provider IDs to assign.

    Returns
    -------
    pd.DataFrame
        Paired dataset with columns from both notes and claims.
    """
    rng = np.random.RandomState(random_state)

    # Map notes to categories
    notes_df = notes_df.copy()
    notes_df["code_category"] = notes_df[specialty_col].map(_map_specialty_to_category)

    # Find primary diagnosis column in claims
    diag_cols = sorted([c for c in claims_df.columns if "DGNS" in c])
    if not diag_cols:
        raise ValueError("No diagnosis columns found in claims data.")
    primary_diag_col = diag_cols[0]

    claims_df = claims_df.copy()
    claims_df["code_category"] = claims_df[primary_diag_col].map(_map_icd9_to_category)
    claims_df = claims_df[claims_df["code_category"] != "unknown"]

    # Pair by category
    paired_rows = []
    categories = notes_df["code_category"].unique()

    for cat in categories:
        cat_notes = notes_df[notes_df["code_category"] == cat]
        cat_claims = claims_df[claims_df["code_category"] == cat]

        if len(cat_notes) == 0 or len(cat_claims) == 0:
            logger.debug("Skipping category '%s' — no data on one side.", cat)
            continue

        # Sample to match sizes
        n_pairs = max(len(cat_notes), len(cat_claims))
        note_indices = rng.choice(cat_notes.index, size=n_pairs, replace=True)
        claim_indices = rng.choice(cat_claims.index, size=n_pairs, replace=True)

        for ni, ci in zip(note_indices, claim_indices):
            note_row = cat_notes.loc[ni]
            claim_row = cat_claims.loc[ci]
            paired_rows.append({
                "pair_id": f"PAIR_{len(paired_rows):06d}",
                "note_id": note_row.get("note_id", ""),
                "transcription": note_row.get("transcription", ""),
                "medical_specialty": note_row.get("medical_specialty", ""),
                "code_category": cat,
                "primary_diagnosis_code": str(claim_row.get(primary_diag_col, "")),
                "claim_id": str(claim_row.get("CLM_ID", claim_row.get("clm_id", ""))),
                "claim_payment": float(claim_row.get("CLM_PMT_AMT", claim_row.get("clm_pmt_amt", 0))),
            })

            # Limit total pairs for manageable dataset size
            if len(paired_rows) >= 10000:
                break
        if len(paired_rows) >= 10000:
            break

    paired_df = pd.DataFrame(paired_rows)

    # Assign synthetic provider IDs
    provider_ids = [f"PRV_{i:04d}" for i in range(n_providers)]
    paired_df["provider_id"] = rng.choice(provider_ids, size=len(paired_df))

    logger.info(
        "Created %d note-claim pairs across %d categories, %d providers",
        len(paired_df),
        paired_df["code_category"].nunique(),
        n_providers,
    )
    return paired_df
