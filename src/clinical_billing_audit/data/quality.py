"""Reusable data-quality checks for clinical notes and claims.

Design
------
Each check is a standalone function that returns a ``QualityResult`` with a
pass/fail flag, a human-readable message, and an optional DataFrame of
flagged rows.  The ``DataQualityChecker`` class orchestrates multiple checks
and produces a summary report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QualityResult:
    """Result of a single data-quality check."""

    check_name: str
    passed: bool
    message: str
    flagged_rows: pd.DataFrame | None = None
    metric_value: float | None = None


@dataclass
class QualityReport:
    """Aggregated results from all quality checks."""

    results: list[QualityResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def summary_df(self) -> pd.DataFrame:
        """Return a summary DataFrame."""
        return pd.DataFrame(
            [
                {
                    "check": r.check_name,
                    "passed": r.passed,
                    "metric": r.metric_value,
                    "message": r.message,
                }
                for r in self.results
            ]
        )

    def save(self, path: str | Path) -> None:
        """Save the report to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_df().to_csv(path, index=False)
        logger.info("Quality report saved to %s", path)


# ── Individual Checks ───────────────────────────────────────


def check_null_fraction(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    max_fraction: float = 0.10,
) -> QualityResult:
    """Check that no column exceeds a maximum null fraction.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    columns : list[str] | None
        Columns to check (default: all).
    max_fraction : float
        Maximum allowable null fraction per column.
    """
    columns = columns or list(df.columns)
    null_fractions = df[columns].isna().mean()
    bad_cols = null_fractions[null_fractions > max_fraction]

    if bad_cols.empty:
        return QualityResult(
            check_name="null_fraction",
            passed=True,
            message=f"All columns have ≤{max_fraction:.0%} nulls.",
            metric_value=float(null_fractions.max()),
        )

    msg = "Columns exceeding null threshold: " + ", ".join(
        f"{col} ({frac:.1%})" for col, frac in bad_cols.items()
    )
    return QualityResult(
        check_name="null_fraction",
        passed=False,
        message=msg,
        metric_value=float(null_fractions.max()),
    )


def check_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    max_fraction: float = 0.05,
) -> QualityResult:
    """Check that duplicate rows don't exceed a threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    subset : list[str] | None
        Columns to use for duplicate detection.
    max_fraction : float
        Maximum allowable duplicate fraction.
    """
    dupes = df.duplicated(subset=subset, keep="first")
    frac = dupes.mean()

    if frac <= max_fraction:
        return QualityResult(
            check_name="duplicates",
            passed=True,
            message=f"Duplicate rate {frac:.2%} ≤ {max_fraction:.0%}.",
            metric_value=float(frac),
        )

    return QualityResult(
        check_name="duplicates",
        passed=False,
        message=f"Duplicate rate {frac:.2%} exceeds {max_fraction:.0%}.",
        flagged_rows=df[dupes],
        metric_value=float(frac),
    )


def check_length_outliers(
    df: pd.DataFrame,
    text_col: str = "transcription",
    min_length: int = 50,
    max_length: int = 50_000,
    iqr_factor: float = 3.0,
) -> QualityResult:
    """Flag notes with extreme lengths (hard bounds + IQR-based).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    text_col : str
        Text column name.
    min_length, max_length : int
        Hard length boundaries.
    iqr_factor : float
        IQR multiplier for statistical outlier detection.
    """
    if text_col not in df.columns:
        return QualityResult(
            check_name="length_outliers",
            passed=True,
            message=f"Column '{text_col}' not found — skipped.",
        )

    lengths = df[text_col].astype(str).str.len()

    # Hard bounds
    hard_outliers = (lengths < min_length) | (lengths > max_length)

    # IQR-based
    q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr
    iqr_outliers = (lengths < lower) | (lengths > upper)

    combined = hard_outliers | iqr_outliers
    n_outliers = int(combined.sum())
    frac = n_outliers / max(len(df), 1)

    if n_outliers == 0:
        return QualityResult(
            check_name="length_outliers",
            passed=True,
            message="No length outliers detected.",
            metric_value=0.0,
        )

    return QualityResult(
        check_name="length_outliers",
        passed=frac < 0.10,  # Fail if >10% are outliers
        message=f"{n_outliers} notes ({frac:.1%}) flagged as length outliers.",
        flagged_rows=df[combined],
        metric_value=float(frac),
    )


def check_class_balance(
    df: pd.DataFrame,
    label_col: str = "medical_specialty",
    min_samples_per_class: int = 5,
) -> QualityResult:
    """Check that each class has at least *min_samples_per_class* rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    label_col : str
        Label column name.
    min_samples_per_class : int
        Minimum acceptable samples per class.
    """
    if label_col not in df.columns:
        return QualityResult(
            check_name="class_balance",
            passed=True,
            message=f"Column '{label_col}' not found — skipped.",
        )

    counts = df[label_col].value_counts()
    small_classes = counts[counts < min_samples_per_class]

    if small_classes.empty:
        return QualityResult(
            check_name="class_balance",
            passed=True,
            message=f"All {len(counts)} classes have ≥{min_samples_per_class} samples.",
            metric_value=float(counts.min()),
        )

    msg = f"{len(small_classes)} classes have <{min_samples_per_class} samples: "
    msg += ", ".join(f"{cls} ({cnt})" for cls, cnt in small_classes.items())
    return QualityResult(
        check_name="class_balance",
        passed=False,
        message=msg,
        metric_value=float(counts.min()),
    )


def check_missing_codes(
    df: pd.DataFrame,
    code_columns: list[str] | None = None,
) -> QualityResult:
    """Check that claims have at least one non-empty code.

    Parameters
    ----------
    df : pd.DataFrame
        Claims DataFrame.
    code_columns : list[str] | None
        Columns containing diagnosis/procedure codes.
    """
    if code_columns is None:
        code_columns = [c for c in df.columns if "DGNS" in c or "PRCDR" in c or "HCPCS" in c]

    if not code_columns:
        return QualityResult(
            check_name="missing_codes",
            passed=True,
            message="No code columns found — skipped.",
        )

    has_code = df[code_columns].apply(
        lambda row: any(str(v).strip() not in ("", "nan", "NaN") for v in row), axis=1
    )
    n_missing = int((~has_code).sum())
    frac = n_missing / max(len(df), 1)

    if n_missing == 0:
        return QualityResult(
            check_name="missing_codes",
            passed=True,
            message="All claims have at least one code.",
            metric_value=0.0,
        )

    return QualityResult(
        check_name="missing_codes",
        passed=frac < 0.05,
        message=f"{n_missing} claims ({frac:.1%}) have no codes.",
        flagged_rows=df[~has_code],
        metric_value=float(frac),
    )


# ── Orchestrator ────────────────────────────────────────────


class DataQualityChecker:
    """Run a suite of quality checks on a DataFrame.

    Example
    -------
    >>> checker = DataQualityChecker(max_null_fraction=0.1)
    >>> report = checker.run(df, dataset_type="notes")
    >>> print(report.summary_df())
    """

    def __init__(
        self,
        max_null_fraction: float = 0.10,
        max_duplicate_fraction: float = 0.05,
        min_note_length: int = 50,
        max_note_length: int = 50_000,
        length_outlier_iqr_factor: float = 3.0,
        min_samples_per_class: int = 5,
    ) -> None:
        self.max_null_fraction = max_null_fraction
        self.max_duplicate_fraction = max_duplicate_fraction
        self.min_note_length = min_note_length
        self.max_note_length = max_note_length
        self.length_outlier_iqr_factor = length_outlier_iqr_factor
        self.min_samples_per_class = min_samples_per_class

    def run(
        self,
        df: pd.DataFrame,
        dataset_type: str = "notes",
        text_col: str = "transcription",
        label_col: str = "medical_specialty",
    ) -> QualityReport:
        """Execute all applicable quality checks.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        dataset_type : str
            ``'notes'`` or ``'claims'`` — determines which checks to run.
        text_col : str
            Text column name (for notes).
        label_col : str
            Label column name (for notes).

        Returns
        -------
        QualityReport
            Aggregated results.
        """
        report = QualityReport()

        # Universal checks
        report.results.append(check_null_fraction(df, max_fraction=self.max_null_fraction))
        report.results.append(check_duplicates(df, max_fraction=self.max_duplicate_fraction))

        if dataset_type == "notes":
            report.results.append(
                check_duplicates(
                    df,
                    subset=[text_col] if text_col in df.columns else None,
                    max_fraction=self.max_duplicate_fraction,
                )
            )
            report.results.append(
                check_length_outliers(
                    df,
                    text_col=text_col,
                    min_length=self.min_note_length,
                    max_length=self.max_note_length,
                    iqr_factor=self.length_outlier_iqr_factor,
                )
            )
            report.results.append(
                check_class_balance(
                    df,
                    label_col=label_col,
                    min_samples_per_class=self.min_samples_per_class,
                )
            )
        elif dataset_type == "claims":
            report.results.append(check_missing_codes(df))

        # Log summary
        logger.info(
            "Quality checks: %d passed, %d failed",
            report.n_passed,
            report.n_failed,
        )
        for r in report.results:
            log_fn = logger.info if r.passed else logger.warning
            log_fn("  [%s] %s — %s", "✓" if r.passed else "✗", r.check_name, r.message)

        return report
