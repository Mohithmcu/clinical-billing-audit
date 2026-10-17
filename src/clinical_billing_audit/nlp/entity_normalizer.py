"""Entity normalisation — map extracted entity text to standardised concepts.

Provides lightweight mapping from free-text entity strings to UMLS-like
concept identifiers without requiring a full UMLS installation.
"""

from __future__ import annotations

import re
from functools import lru_cache

from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)

# ── Diagnosis Normalization Map ─────────────────────────────
_DIAGNOSIS_NORM: dict[str, str] = {
    # Cardiovascular
    "hypertension": "essential_hypertension",
    "high blood pressure": "essential_hypertension",
    "htn": "essential_hypertension",
    "coronary artery disease": "coronary_atherosclerosis",
    "cad": "coronary_atherosclerosis",
    "heart failure": "congestive_heart_failure",
    "chf": "congestive_heart_failure",
    "congestive heart failure": "congestive_heart_failure",
    "atrial fibrillation": "atrial_fibrillation",
    "afib": "atrial_fibrillation",
    "myocardial infarction": "myocardial_infarction",
    "heart attack": "myocardial_infarction",
    "stroke": "cerebrovascular_accident",
    "cva": "cerebrovascular_accident",
    # Metabolic / Endocrine
    "diabetes": "diabetes_mellitus_type2",
    "diabetes mellitus": "diabetes_mellitus_type2",
    "type 2 diabetes": "diabetes_mellitus_type2",
    "type 2 diabetes mellitus": "diabetes_mellitus_type2",
    "dm": "diabetes_mellitus_type2",
    "type 1 diabetes": "diabetes_mellitus_type1",
    "hypothyroidism": "hypothyroidism",
    "hyperthyroidism": "hyperthyroidism",
    "obesity": "obesity",
    # Respiratory
    "asthma": "asthma",
    "copd": "copd",
    "chronic obstructive pulmonary disease": "copd",
    "pneumonia": "pneumonia",
    "pulmonary embolism": "pulmonary_embolism",
    # Musculoskeletal
    "arthritis": "arthritis",
    "osteoarthritis": "osteoarthritis",
    "rheumatoid arthritis": "rheumatoid_arthritis",
    "osteoporosis": "osteoporosis",
    "fracture": "fracture",
    # Neurological
    "epilepsy": "epilepsy",
    "migraine": "migraine",
    "alzheimer": "alzheimer_disease",
    "parkinson": "parkinson_disease",
    # Mental health
    "depression": "depressive_disorder",
    "anxiety": "anxiety_disorder",
    "schizophrenia": "schizophrenia",
    # GI
    "gastroesophageal reflux": "gerd",
    "gerd": "gerd",
    "hepatitis": "hepatitis",
    "cirrhosis": "cirrhosis",
    # Renal
    "chronic kidney disease": "chronic_kidney_disease",
    "ckd": "chronic_kidney_disease",
    "urinary tract infection": "urinary_tract_infection",
    "uti": "urinary_tract_infection",
    # Oncology
    "cancer": "malignant_neoplasm",
    "carcinoma": "malignant_neoplasm",
    "melanoma": "melanoma",
    "lymphoma": "lymphoma",
    "leukemia": "leukemia",
    # Infectious
    "sepsis": "sepsis",
    "cellulitis": "cellulitis",
    # Hematologic
    "anemia": "anemia",
    "deep vein thrombosis": "deep_vein_thrombosis",
    "dvt": "deep_vein_thrombosis",
}

# ── Procedure Normalization Map ─────────────────────────────
_PROCEDURE_NORM: dict[str, str] = {
    "colonoscopy": "GI_endoscopy",
    "endoscopy": "GI_endoscopy",
    "biopsy": "tissue_biopsy",
    "excision": "surgical_excision",
    "resection": "surgical_resection",
    "arthroplasty": "joint_replacement",
    "appendectomy": "appendectomy",
    "cholecystectomy": "cholecystectomy",
    "hysterectomy": "hysterectomy",
    "mastectomy": "mastectomy",
    "laparoscopy": "laparoscopic_surgery",
    "arthroscopy": "arthroscopic_surgery",
    "catheterization": "catheterization",
    "angioplasty": "angioplasty",
    "stent placement": "stent_placement",
    "chemotherapy": "chemotherapy",
    "radiation therapy": "radiation_therapy",
    "dialysis": "dialysis",
    "blood transfusion": "blood_transfusion",
    "ct scan": "CT_imaging",
    "mri": "MRI_imaging",
    "x-ray": "radiography",
    "xray": "radiography",
    "ultrasound": "ultrasonography",
    "echocardiogram": "echocardiography",
    "electrocardiogram": "ECG",
    "ekg": "ECG",
    "ecg": "ECG",
    "physical therapy": "physical_therapy",
    "bypass": "bypass_surgery",
    "transplant": "organ_transplant",
    "craniotomy": "craniotomy",
    "suture": "wound_closure",
    "debridement": "debridement",
}


@lru_cache(maxsize=10000)
def normalize_entity(text: str, entity_type: str = "DISEASE") -> str:
    """Normalise an entity string to a canonical concept.

    Parameters
    ----------
    text : str
        Raw entity text (e.g. ``"heart failure"``).
    entity_type : str
        Entity type: ``DISEASE``, ``CHEMICAL``, ``PROCEDURE``, etc.

    Returns
    -------
    str
        Normalised concept string.
    """
    clean = text.lower().strip()
    clean = re.sub(r"\s+", " ", clean)

    if entity_type == "DISEASE":
        return _DIAGNOSIS_NORM.get(clean, clean.replace(" ", "_"))
    elif entity_type == "PROCEDURE":
        return _PROCEDURE_NORM.get(clean, clean.replace(" ", "_"))
    else:
        # For chemicals and anatomy, just clean up
        return clean.replace(" ", "_")


def normalize_ner_results(
    entities: list[dict],
) -> list[dict]:
    """Normalise a list of entity dictionaries.

    Parameters
    ----------
    entities : list[dict]
        Each dict has ``text`` and ``label`` keys.

    Returns
    -------
    list[dict]
        Same dicts with an added ``normalized`` key.
    """
    for ent in entities:
        ent["normalized"] = normalize_entity(ent["text"], ent.get("label", "DISEASE"))
    return entities
