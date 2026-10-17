"""Map extracted clinical entities to ICD / code categories.

Provides a lightweight mapping from normalised entity concepts to broad
ICD-10 chapter categories and HCPCS category groups, enabling comparison
with billed codes.
"""

from __future__ import annotations

from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)

# ── Diagnosis → ICD-10 Category Mapping ─────────────────────
_DIAGNOSIS_TO_CATEGORY: dict[str, str] = {
    # Cardiovascular
    "essential_hypertension": "circulatory",
    "coronary_atherosclerosis": "circulatory",
    "congestive_heart_failure": "circulatory",
    "atrial_fibrillation": "circulatory",
    "myocardial_infarction": "circulatory",
    "cerebrovascular_accident": "circulatory",
    "deep_vein_thrombosis": "circulatory",
    "pulmonary_embolism": "circulatory",
    # Metabolic / Endocrine
    "diabetes_mellitus_type2": "endocrine",
    "diabetes_mellitus_type1": "endocrine",
    "hypothyroidism": "endocrine",
    "hyperthyroidism": "endocrine",
    "obesity": "endocrine",
    # Respiratory
    "asthma": "respiratory",
    "copd": "respiratory",
    "pneumonia": "respiratory",
    # Musculoskeletal
    "arthritis": "musculoskeletal",
    "osteoarthritis": "musculoskeletal",
    "rheumatoid_arthritis": "musculoskeletal",
    "osteoporosis": "musculoskeletal",
    "fracture": "injury_external",
    # Neurological
    "epilepsy": "nervous",
    "migraine": "nervous",
    "alzheimer_disease": "nervous",
    "parkinson_disease": "nervous",
    # Mental health
    "depressive_disorder": "mental_behavioral",
    "anxiety_disorder": "mental_behavioral",
    "schizophrenia": "mental_behavioral",
    # GI
    "gerd": "digestive",
    "hepatitis": "digestive",
    "cirrhosis": "digestive",
    # Renal
    "chronic_kidney_disease": "genitourinary",
    "urinary_tract_infection": "genitourinary",
    # Oncology
    "malignant_neoplasm": "neoplasms",
    "melanoma": "neoplasms",
    "lymphoma": "neoplasms",
    "leukemia": "neoplasms",
    # Infectious
    "sepsis": "infectious",
    "cellulitis": "skin_subcutaneous",
    # Hematologic
    "anemia": "blood",
}

# ── Procedure → HCPCS Category Mapping ─────────────────────
_PROCEDURE_TO_CATEGORY: dict[str, str] = {
    "GI_endoscopy": "digestive",
    "tissue_biopsy": "diagnostic_imaging",
    "surgical_excision": "surgical",
    "surgical_resection": "surgical",
    "joint_replacement": "musculoskeletal",
    "appendectomy": "digestive",
    "cholecystectomy": "digestive",
    "hysterectomy": "pregnancy_reproductive",
    "mastectomy": "neoplasms",
    "laparoscopic_surgery": "surgical",
    "arthroscopic_surgery": "musculoskeletal",
    "catheterization": "circulatory",
    "angioplasty": "circulatory",
    "stent_placement": "circulatory",
    "chemotherapy": "neoplasms",
    "radiation_therapy": "neoplasms",
    "dialysis": "genitourinary",
    "blood_transfusion": "blood",
    "CT_imaging": "diagnostic_imaging",
    "MRI_imaging": "diagnostic_imaging",
    "radiography": "diagnostic_imaging",
    "ultrasonography": "diagnostic_imaging",
    "echocardiography": "circulatory",
    "ECG": "circulatory",
    "physical_therapy": "musculoskeletal",
    "bypass_surgery": "circulatory",
    "organ_transplant": "surgical",
    "craniotomy": "nervous",
    "wound_closure": "injury_external",
    "debridement": "skin_subcutaneous",
}


def entity_to_code_category(
    normalized_entity: str,
    entity_type: str = "DISEASE",
) -> str:
    """Map a normalised entity to its most likely ICD/HCPCS code category.

    Parameters
    ----------
    normalized_entity : str
        Normalised entity string (from :mod:`entity_normalizer`).
    entity_type : str
        Entity type: ``DISEASE``, ``PROCEDURE``, ``CHEMICAL``, etc.

    Returns
    -------
    str
        Broad code category (e.g. ``'circulatory'``, ``'digestive'``).
    """
    if entity_type in ("DISEASE",):
        return _DIAGNOSIS_TO_CATEGORY.get(normalized_entity, "general")
    elif entity_type in ("PROCEDURE",):
        return _PROCEDURE_TO_CATEGORY.get(normalized_entity, "general")
    else:
        return "general"


def predict_code_categories_from_entities(
    diagnoses: list[str],
    procedures: list[str],
) -> dict[str, float]:
    """Predict likely code categories from extracted entities.

    Uses a voting scheme: each entity maps to a category, and we return
    normalised vote counts.

    Parameters
    ----------
    diagnoses : list[str]
        Normalised diagnosis entity strings.
    procedures : list[str]
        Normalised procedure entity strings.

    Returns
    -------
    dict[str, float]
        Category → confidence score (0–1, sums to 1).
    """
    votes: dict[str, int] = {}

    for diag in diagnoses:
        cat = entity_to_code_category(diag, "DISEASE")
        votes[cat] = votes.get(cat, 0) + 1

    for proc in procedures:
        cat = entity_to_code_category(proc, "PROCEDURE")
        votes[cat] = votes.get(cat, 0) + 1

    total = sum(votes.values())
    if total == 0:
        return {"general": 1.0}

    return {cat: count / total for cat, count in sorted(votes.items(), key=lambda x: -x[1])}


def get_primary_predicted_category(
    diagnoses: list[str],
    procedures: list[str],
) -> str:
    """Return the single most-likely code category.

    Parameters
    ----------
    diagnoses : list[str]
        Normalised diagnosis entities.
    procedures : list[str]
        Normalised procedure entities.

    Returns
    -------
    str
        Most likely code category.
    """
    scores = predict_code_categories_from_entities(diagnoses, procedures)
    return max(scores, key=scores.get) if scores else "general"
