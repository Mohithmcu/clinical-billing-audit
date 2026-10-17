"""Shared pytest fixtures for the clinical_billing_audit test suite."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_notes_df() -> pd.DataFrame:
    """Fixture providing a small DataFrame of clinical notes."""
    return pd.DataFrame(
        {
            "note_id": [f"MT_{i:04d}" for i in range(10)],
            "description": [f"Description for sample {i}" for i in range(10)],
            "medical_specialty": [
                "Cardiovascular / Pulmonary",
                "Cardiovascular / Pulmonary",
                "Orthopedic",
                "Orthopedic",
                "Gastroenterology",
                "Gastroenterology",
                "Neurology",
                "Neurology",
                "Surgery",
                "Urology",
            ],
            "sample_name": [f"Sample {i}" for i in range(10)],
            "transcription": [
                "Patient is a 64-year-old male with severe chest pain, coronary artery disease, and hypertension. Started on lisinopril and aspirin.",
                "Decompensated congestive heart failure with dyspnea on exertion and bilateral pedal edema. Administered IV furosemide.",
                "Left knee joint osteoarthritis with medial meniscus tear. Scheduled for diagnostic arthroscopy and partial meniscectomy.",
                "Right rotator cuff tear of the supraspinatus tendon. Recommend physical therapy and subacromial decompression.",
                "Routine screening colonoscopy. Moderate diverticulosis coli identified; 1 cm polyp resected via snare polypectomy.",
                "Gastroesophageal reflux disease with grade B erosive esophagitis. Prescribed omeprazole 40 mg daily.",
                "Severe migraine headache with visual aura and nausea, refractory to sumatriptan. Commenced topiramate prophylaxis.",
                "Transient ischemic attack with right-sided weakness. Carotid ultrasound shows moderate bifurcation stenosis. Commenced clopidogrel.",
                "Laparoscopic appendectomy performed for acute gangrenous appendicitis. Specimen sent to pathology. Recovery uneventful.",
                "Obstructing ureteral calculus with severe flank pain and microscopic hematuria. Double-J ureteral stent inserted via cystoscopy.",
            ],
            "keywords": [
                "cardiac, pain",
                "chf",
                "knee, tear",
                "shoulder",
                "colon",
                "gerd",
                "headache",
                "tia",
                "appendix",
                "stone",
            ],
        }
    )


@pytest.fixture
def sample_claims_df() -> pd.DataFrame:
    """Fixture providing a small DataFrame of CMS claims."""
    return pd.DataFrame(
        {
            "DESYNPUF_ID": [f"BENE_{i:05d}" for i in range(10)],
            "CLM_ID": [f"CLM_{i:07d}" for i in range(10)],
            "SEGMENT": [1] * 10,
            "CLM_FROM_DT": [20080101 + i for i in range(10)],
            "CLM_THRU_DT": [20080105 + i for i in range(10)],
            "PRVDR_NUM": [f"PRV_{i % 3:04d}" for i in range(10)],
            "CLM_PMT_AMT": [1200.0 + i * 250 for i in range(10)],
            "NCH_PRMRY_PYR_CLM_PD_AMT": [0.0] * 10,
            "AT_PHYSN_NPI": [f"NPI_{1000 + i}" for i in range(10)],
            "OP_PHYSN_NPI": [f"NPI_{2000 + i}" for i in range(10)],
            "OT_PHYSN_NPI": [""] * 10,
            "CLM_ADMSN_DT": [20080101 + i for i in range(10)],
            "ADMTNG_ICD9_DGNS_CD": [
                "4019",
                "4280",
                "71590",
                "72761",
                "2113",
                "53081",
                "34690",
                "4359",
                "5409",
                "5921",
            ],
            "CLM_PASS_THRU_PER_DIEM_AMT": [0.0] * 10,
            "NCH_BENE_IP_DDCTBL_AMT": [1024.0] * 10,
            "NCH_BENE_PTA_COINSRNC_LBLTY_AM": [0.0] * 10,
            "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM": [0.0] * 10,
            "CLM_UTLZTN_AMT": [1500.0 + i * 300 for i in range(10)],
            "PRVDR_STATE_CD": ["CA", "NY", "TX", "FL", "IL", "CA", "NY", "TX", "FL", "IL"],
            "ICD9_DGNS_CD_1": [
                "4019",
                "4280",
                "71590",
                "72761",
                "2113",
                "53081",
                "34690",
                "4359",
                "5409",
                "5921",
            ],
            "ICD9_DGNS_CD_2": [
                "41401",
                "4019",
                "7140",
                "72610",
                "56210",
                "53010",
                "7840",
                "4019",
                "78900",
                "591",
            ],
            "ICD9_PRCDR_CD_1": [
                "3606",
                "8872",
                "8154",
                "8363",
                "4523",
                "4513",
                "8703",
                "8871",
                "4709",
                "598",
            ],
        }
    )


@pytest.fixture
def sample_paired_df(sample_notes_df: pd.DataFrame) -> pd.DataFrame:
    """Fixture providing a small paired DataFrame ready for modeling."""
    df = sample_notes_df.copy()
    df["pair_id"] = [f"PAIR_{i:04d}" for i in range(len(df))]
    df["claim_id"] = [f"CLM_{i:07d}" for i in range(len(df))]
    df["provider_id"] = [f"PRV_{i % 3:04d}" for i in range(len(df))]
    df["claim_payment"] = [1500.0 + i * 400.0 for i in range(len(df))]
    df["code_category"] = [
        "circulatory",
        "circulatory",
        "musculoskeletal",
        "musculoskeletal",
        "digestive",
        "digestive",
        "nervous",
        "nervous",
        "surgical",
        "genitourinary",
    ]
    df["primary_diagnosis_code"] = [
        "4019",
        "4280",
        "71590",
        "72761",
        "2113",
        "53081",
        "34690",
        "4359",
        "5409",
        "5921",
    ]
    df["is_injected_mismatch"] = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    df["mismatch_type"] = [
        "clean",
        "upcoding",
        "clean",
        "clean",
        "miscoding",
        "clean",
        "clean",
        "undercoding",
        "clean",
        "clean",
    ]
    df["original_code_category"] = df["code_category"].copy()
    return df


@pytest.fixture
def temp_dir() -> Path:
    """Fixture providing a temporary directory."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)
