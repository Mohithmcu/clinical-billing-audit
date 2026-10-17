"""Unit tests for NER extraction, entity normalisation, and code mapping."""

from __future__ import annotations

import pandas as pd
import pytest

from clinical_billing_audit.nlp.code_mapper import (
    entity_to_code_category,
    get_primary_predicted_category,
    predict_code_categories_from_entities,
)
from clinical_billing_audit.nlp.entity_normalizer import (
    normalize_entity,
    normalize_ner_results,
)
from clinical_billing_audit.nlp.ner_extractor import (
    ClinicalEntity,
    NERExtractor,
    NERResult,
)
from clinical_billing_audit.nlp.text_preprocessor import preprocess_note


def test_clinical_entity_and_ner_result():
    """Test ClinicalEntity and NERResult dataclasses."""
    e1 = ClinicalEntity(text="hypertension", label="DISEASE", start=0, end=12)
    e2 = ClinicalEntity(text="lisinopril", label="CHEMICAL", start=13, end=23)
    e3 = ClinicalEntity(text="colonoscopy", label="PROCEDURE", start=24, end=35)
    e4 = ClinicalEntity(text="knee", label="ANATOMY", start=36, end=40)

    result = NERResult(note_id="MT_0001", entities=[e1, e2, e3, e4])
    assert result.note_id == "MT_0001"
    assert result.diagnoses == ["hypertension"]
    assert result.medications == ["lisinopril"]
    assert result.procedures == ["colonoscopy"]
    assert result.anatomy == ["knee"]
    assert result.to_dict()["n_entities"] == 4


def test_preprocess_note():
    """Test text preprocessing rules."""
    text = "Patient with CAD and HTN. Status post MI. Follow up [**Hospital 123**]."
    cleaned = preprocess_note(text)
    assert "[REDACTED]" in cleaned or "[**" not in cleaned


def test_extract_entities_from_note():
    """Test entity extraction from a clinical note."""
    extractor = NERExtractor()
    note = "Patient has severe essential hypertension and congestive heart failure. Prescribed furosemide and lisinopril. Underwent diagnostic colonoscopy."
    result = extractor.extract(note, note_id="TEST_01")
    assert isinstance(result, NERResult)
    assert result.note_id == "TEST_01"
    assert len(result.entities) > 0


def test_extract_entities_batch(sample_notes_df: pd.DataFrame):
    """Test batch entity extraction returns a DataFrame."""
    extractor = NERExtractor()
    results = extractor.batch_extract(sample_notes_df)
    assert len(results) == len(sample_notes_df)
    results_df = NERExtractor.results_to_dataframe(results)
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == len(sample_notes_df)
    assert "note_id" in results_df.columns
    assert "n_diagnoses" in results_df.columns
    assert "n_medications" in results_df.columns


def test_normalize_entity():
    """Test entity normalisation rules."""
    assert normalize_entity("hypertension", "DISEASE") == "essential_hypertension"
    assert normalize_entity("type 2 diabetes", "DISEASE") == "diabetes_mellitus_type2"
    assert normalize_entity("colonoscopy", "PROCEDURE") == "GI_endoscopy"

    ents = [{"text": "hypertension", "label": "DISEASE"}]
    norm_ents = normalize_ner_results(ents)
    assert norm_ents[0]["normalized"] == "essential_hypertension"


def test_map_entities_to_categories():
    """Test mapping clinical entities to ICD code categories."""
    diag_cat = entity_to_code_category("essential_hypertension", "DISEASE")
    assert diag_cat == "circulatory"

    preds = predict_code_categories_from_entities(
        diagnoses=["essential_hypertension"],
        procedures=["GI_endoscopy"],
    )
    assert "circulatory" in preds or "digestive" in preds

    primary = get_primary_predicted_category(
        diagnoses=["essential_hypertension"],
        procedures=[],
    )
    assert primary == "circulatory"
