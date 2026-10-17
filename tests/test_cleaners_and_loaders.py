"""Unit tests for cleaners, loaders, pairing, and mismatch injector modules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from clinical_billing_audit.data.cleaners import (
    clean_claims_data,
    clean_clinical_notes,
    clean_specialty_labels,
    profile_dataset,
)
from clinical_billing_audit.data.loaders import (
    load_cms_claims,
    load_icd9_mapping,
    load_mtsamples,
)
from clinical_billing_audit.data.mismatch_injector import inject_mismatches
from clinical_billing_audit.data.pairing import pair_notes_and_claims


def test_clean_clinical_notes():
    """Test clinical note cleaning pipeline."""
    raw_df = pd.DataFrame({
        "transcription": [
            "  Valid note text that has plenty of length to exceed fifty characters easily.  ",
            "Short",  # Too short
            None,     # Null
            "  Valid note text that has plenty of length to exceed fifty characters easily.  ",  # Duplicate
        ]
    })
    cleaned = clean_clinical_notes(raw_df, min_length=50)
    assert len(cleaned) == 1
    assert "Valid note text" in cleaned["transcription"].iloc[0]


def test_clean_specialty_labels():
    """Test specialty label cleaning."""
    df = pd.DataFrame({
        "medical_specialty": ["  Cardiovascular / Pulmonary  ", "Orthopedic -", "nan", None]
    })
    cleaned = clean_specialty_labels(df)
    assert len(cleaned) == 2
    assert "Cardiovascular / Pulmonary" in cleaned["medical_specialty"].values
    assert "Orthopedic" in cleaned["medical_specialty"].values


def test_profile_dataset(sample_notes_df: pd.DataFrame, temp_dir: Path):
    """Test statistical profiling on dataset."""
    out_csv = temp_dir / "profile.csv"
    profile = profile_dataset(sample_notes_df, output_path=out_csv)
    assert profile["n_rows"] == 10
    assert "text_length" in profile
    assert profile["text_length"]["mean"] > 0
    assert out_csv.exists()


def test_clean_claims_data(sample_claims_df: pd.DataFrame):
    """Test CMS claims data cleaning."""
    cleaned = clean_claims_data(sample_claims_df)
    assert len(cleaned) == 10
    assert "ICD9_DGNS_CD_1" in cleaned.columns
    assert "CLM_PMT_AMT" in cleaned.columns
    assert "PRVDR_NUM" in cleaned.columns


def test_load_icd9_mapping():
    """Test ICD-9 mapping dictionary."""
    mapping = load_icd9_mapping()
    assert isinstance(mapping, dict)
    assert "4019" in mapping
    assert "Essential hypertension" in mapping["4019"]


def test_pair_notes_and_claims(sample_notes_df: pd.DataFrame, sample_claims_df: pd.DataFrame):
    """Test pairing of clinical notes with synthetic claims."""
    cleaned_claims = clean_claims_data(sample_claims_df)
    paired = pair_notes_and_claims(sample_notes_df, cleaned_claims, random_state=42)
    assert len(paired) > 0
    assert "code_category" in paired.columns
    assert "pair_id" in paired.columns
    assert "provider_id" in paired.columns


def test_inject_mismatches(sample_paired_df: pd.DataFrame):
    """Test synthetic mismatch injection."""
    injected = inject_mismatches(sample_paired_df, injection_rate=0.4, random_state=42)
    assert "is_mismatch" in injected.columns
    assert "mismatch_type" in injected.columns
    assert "billed_category" in injected.columns
    # Check that some mismatches were injected
    assert (injected["is_mismatch"] == True).sum() > 0
