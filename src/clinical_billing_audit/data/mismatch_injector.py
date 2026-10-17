"""Inject synthetic billing mismatches for FWA detection evaluation.

Mismatch Types
--------------
- **Upcoding**: Replace the billed code category with a more complex/expensive
  one while the note text stays the same.
- **Miscoding**: Replace the billed code category with a completely unrelated one.
- **Undercoding**: Replace the billed code with a less complex/expensive one.

Every injected mismatch is labelled so we have ground truth for evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clinical_billing_audit.utils.constants import MISMATCH_LABELS
from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)

# Categories ranked roughly by average cost/complexity
_COMPLEXITY_RANK = [
    "general",
    "dental",
    "immune",
    "skin_subcutaneous",
    "ear_mastoid",
    "eye_adnexa",
    "mental_behavioral",
    "endocrine",
    "respiratory",
    "digestive",
    "genitourinary",
    "musculoskeletal",
    "nervous",
    "circulatory",
    "neoplasms",
    "injury_external",
    "surgical",
    "pregnancy_reproductive",
    "diagnostic_imaging",
]

_RANK_MAP = {cat: i for i, cat in enumerate(_COMPLEXITY_RANK)}


def _get_upcode(category: str, rng: np.random.RandomState) -> str:
    """Return a higher-complexity category than *category*."""
    rank = _RANK_MAP.get(category, len(_COMPLEXITY_RANK) // 2)
    higher = [c for c, r in _RANK_MAP.items() if r > rank]
    if not higher:
        higher = [c for c in _COMPLEXITY_RANK if c != category]
    return rng.choice(higher)


def _get_miscode(category: str, rng: np.random.RandomState) -> str:
    """Return a completely unrelated category."""
    others = [c for c in _COMPLEXITY_RANK if c != category]
    return rng.choice(others)


def _get_undercode(category: str, rng: np.random.RandomState) -> str:
    """Return a lower-complexity category than *category*."""
    rank = _RANK_MAP.get(category, len(_COMPLEXITY_RANK) // 2)
    lower = [c for c, r in _RANK_MAP.items() if r < rank]
    if not lower:
        lower = [c for c in _COMPLEXITY_RANK if c != category]
    return rng.choice(lower)


def inject_mismatches(
    df: pd.DataFrame,
    injection_rate: float = 0.20,
    mismatch_types: list[str] | None = None,
    mismatch_weights: list[float] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Inject synthetic billing mismatches into a paired note-claim dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Paired dataset with ``code_category`` column.
    injection_rate : float
        Fraction of rows to corrupt (default 20%).
    mismatch_types : list[str] | None
        Types of mismatches to inject.
    mismatch_weights : list[float] | None
        Probability weights for each mismatch type.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Dataset with added columns:
        - ``billed_category``: the (possibly corrupted) billed category
        - ``mismatch_type``: label (correct_match / upcoding / miscoding / undercoding)
        - ``mismatch_label``: integer label (0=correct, 1=upcoding, 2=miscoding, 3=undercoding)
        - ``is_mismatch``: boolean flag
    """
    if mismatch_types is None:
        mismatch_types = ["upcoding", "miscoding", "undercoding"]
    if mismatch_weights is None:
        mismatch_weights = [0.5, 0.3, 0.2]

    rng = np.random.RandomState(random_state)
    df = df.copy()

    n_total = len(df)
    n_mismatches = int(n_total * injection_rate)

    # Select rows to corrupt
    mismatch_indices = rng.choice(df.index, size=n_mismatches, replace=False)

    # Initialise columns
    df["billed_category"] = df["code_category"]  # Start with correct
    df["mismatch_type"] = "correct_match"
    df["mismatch_label"] = MISMATCH_LABELS["correct_match"]
    df["is_mismatch"] = False

    # Assign mismatch types
    weights = np.array(mismatch_weights, dtype=float)
    weights /= weights.sum()
    assigned_types = rng.choice(mismatch_types, size=n_mismatches, p=weights)

    generators = {
        "upcoding": _get_upcode,
        "miscoding": _get_miscode,
        "undercoding": _get_undercode,
    }

    for idx, mtype in zip(mismatch_indices, assigned_types):
        original_cat = df.at[idx, "code_category"]
        generator = generators.get(mtype, _get_miscode)
        new_cat = generator(original_cat, rng)

        df.at[idx, "billed_category"] = new_cat
        df.at[idx, "mismatch_type"] = mtype
        df.at[idx, "mismatch_label"] = MISMATCH_LABELS.get(mtype, 2)
        df.at[idx, "is_mismatch"] = True

    # Summary
    type_counts = df["mismatch_type"].value_counts()
    logger.info(
        "Injected %d mismatches (%.1f%%) into %d pairs:",
        n_mismatches, 100 * injection_rate, n_total,
    )
    for mtype, count in type_counts.items():
        logger.info("  %s: %d (%.1f%%)", mtype, count, 100 * count / n_total)

    return df
