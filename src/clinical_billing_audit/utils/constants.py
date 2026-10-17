"""Constants used across the clinical-billing-audit pipeline."""

from pathlib import Path

# ── Project Root ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── Directory Defaults ──────────────────────────────────────
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_HAND_LABELED_DIR = PROJECT_ROOT / "data" / "hand_labeled"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

# ── Specialty Mapping ───────────────────────────────────────
# Map MTSamples specialty labels → broad code categories used for pairing
SPECIALTY_TO_CODE_CATEGORY = {
    "Cardiovascular / Pulmonary": "circulatory",
    "Orthopedic": "musculoskeletal",
    "Gastroenterology": "digestive",
    "Neurology": "nervous",
    "Urology": "genitourinary",
    "Obstetrics / Gynecology": "pregnancy_reproductive",
    "Ophthalmology": "eye_adnexa",
    "ENT - Otolaryngology": "ear_mastoid",
    "Dermatology": "skin_subcutaneous",
    "Hematology - Oncology": "neoplasms",
    "Nephrology": "genitourinary",
    "Endocrinology": "endocrine",
    "Psychiatry / Psychology": "mental_behavioral",
    "Radiology": "diagnostic_imaging",
    "General Medicine": "general",
    "Allergy / Immunology": "immune",
    "Rheumatology": "musculoskeletal",
    "Pulmonary": "circulatory",
    "Pediatrics": "general",
    "Surgery": "surgical",
    "Emergency Room Reports": "injury_external",
    "Dentistry": "dental",
    "Pain Management": "nervous",
    "Physical Medicine - Rehab": "musculoskeletal",
    "Podiatry": "musculoskeletal",
    "Cosmetic / Plastic Surgery": "skin_subcutaneous",
    "Bariatrics": "digestive",
    "Sleep Medicine": "nervous",
    "Hospice - Palliative Care": "general",
    "IME-QME-Work Comp etc.": "injury_external",
    "Chiropractic": "musculoskeletal",
    "Autopsy": "general",
    "Discharge Summary": "general",
    "Consult - History and Phy.": "general",
    "Lab Medicine - Pathology": "diagnostic_imaging",
    "Letters": "general",
    "SOAP / Chart / Progress Notes": "general",
    "Office Notes": "general",
    "Speech - Language": "nervous",
    "Diets and Nutritions": "endocrine",
}

# ── ICD-10 Chapter Ranges (simplified) ─────────────────────
ICD10_CHAPTERS = {
    "infectious": ("A00", "B99"),
    "neoplasms": ("C00", "D49"),
    "blood": ("D50", "D89"),
    "endocrine": ("E00", "E89"),
    "mental_behavioral": ("F01", "F99"),
    "nervous": ("G00", "G99"),
    "eye_adnexa": ("H00", "H59"),
    "ear_mastoid": ("H60", "H95"),
    "circulatory": ("I00", "I99"),
    "respiratory": ("J00", "J99"),
    "digestive": ("K00", "K95"),
    "skin_subcutaneous": ("L00", "L99"),
    "musculoskeletal": ("M00", "M99"),
    "genitourinary": ("N00", "N99"),
    "pregnancy_reproductive": ("O00", "O9A"),
    "perinatal": ("P00", "P96"),
    "congenital": ("Q00", "Q99"),
    "symptoms_abnormal": ("R00", "R99"),
    "injury_external": ("S00", "T88"),
    "external_causes": ("V00", "Y99"),
    "health_services": ("Z00", "Z99"),
}

# ── CMS SynPUF Column Mappings ─────────────────────────────
CMS_CLAIM_COLUMNS = {
    "beneficiary_id": "DESYNPUF_ID",
    "claim_id": "CLM_ID",
    "diagnosis_1": "ICD9_DGNS_CD_1",
    "diagnosis_2": "ICD9_DGNS_CD_2",
    "diagnosis_3": "ICD9_DGNS_CD_3",
    "diagnosis_4": "ICD9_DGNS_CD_4",
    "diagnosis_5": "ICD9_DGNS_CD_5",
    "diagnosis_6": "ICD9_DGNS_CD_6",
    "diagnosis_7": "ICD9_DGNS_CD_7",
    "diagnosis_8": "ICD9_DGNS_CD_8",
    "diagnosis_9": "ICD9_DGNS_CD_9",
    "diagnosis_10": "ICD9_DGNS_CD_10",
    "procedure_1": "ICD9_PRCDR_CD_1",
    "procedure_2": "ICD9_PRCDR_CD_2",
    "procedure_3": "ICD9_PRCDR_CD_3",
    "procedure_4": "ICD9_PRCDR_CD_4",
    "procedure_5": "ICD9_PRCDR_CD_5",
    "procedure_6": "ICD9_PRCDR_CD_6",
    "hcpcs_1": "HCPCS_CD_1",
    "hcpcs_2": "HCPCS_CD_2",
    "hcpcs_3": "HCPCS_CD_3",
    "provider_id": "PRVDR_NUM",
    "claim_payment": "CLM_PMT_AMT",
    "claim_from_date": "CLM_FROM_DT",
    "claim_thru_date": "CLM_THRU_DT",
}

# ── Mismatch Types ──────────────────────────────────────────
MISMATCH_LABELS = {
    "correct_match": 0,
    "upcoding": 1,
    "miscoding": 2,
    "undercoding": 3,
}

MISMATCH_LABEL_NAMES = {v: k for k, v in MISMATCH_LABELS.items()}

# ── Anomaly Severity Levels ────────────────────────────────
SEVERITY_LEVELS = {
    "HIGH_RISK": "Likely Upcoding",
    "MEDIUM_RISK": "Possible Miscoding",
    "LOW_RISK": "Unusual Pattern",
    "NORMAL": "Normal",
}

# ── ICD-9 to Broad Category Mapping (for CMS SynPUF) ──────
# CMS SynPUF uses ICD-9 codes; map first digit to broad category
ICD9_BROAD_CATEGORY = {
    "0": "infectious",
    "1": "neoplasms",
    "2": "endocrine",
    "3": "blood",
    "4": "mental_behavioral",
    "5": "digestive",
    "6": "genitourinary",
    "7": "musculoskeletal",
    "8": "injury_external",
    "9": "injury_external",
    "E": "external_causes",
    "V": "health_services",
}

# ── Common Clinical Abbreviations ──────────────────────────
CLINICAL_ABBREVIATIONS = {
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "dm2": "type 2 diabetes mellitus",
    "cad": "coronary artery disease",
    "chf": "congestive heart failure",
    "copd": "chronic obstructive pulmonary disease",
    "ckd": "chronic kidney disease",
    "afib": "atrial fibrillation",
    "mi": "myocardial infarction",
    "cva": "cerebrovascular accident",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "uti": "urinary tract infection",
    "gerd": "gastroesophageal reflux disease",
    "bph": "benign prostatic hyperplasia",
    "oa": "osteoarthritis",
    "ra": "rheumatoid arthritis",
    "sle": "systemic lupus erythematosus",
    "tia": "transient ischemic attack",
    "ards": "acute respiratory distress syndrome",
}
