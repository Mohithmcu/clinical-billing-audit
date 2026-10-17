"""Biomedical Named Entity Recognition (NER) using scispaCy.

Extracts diagnoses, medications, procedures, and anatomy entities from
clinical notes.  Falls back to a lightweight regex-based extractor when
scispaCy models are not installed (e.g., in CI or lightweight environments).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from clinical_billing_audit.nlp.text_preprocessor import preprocess_note
from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ClinicalEntity:
    """A single extracted clinical entity."""

    text: str
    label: str  # e.g. DISEASE, CHEMICAL, PROCEDURE
    start: int = 0
    end: int = 0
    cui: str | None = None  # UMLS CUI if available
    confidence: float = 1.0


@dataclass
class NERResult:
    """NER extraction result for a single note."""

    note_id: str
    entities: list[ClinicalEntity] = field(default_factory=list)

    @property
    def diagnoses(self) -> list[str]:
        return [e.text for e in self.entities if e.label == "DISEASE"]

    @property
    def medications(self) -> list[str]:
        return [e.text for e in self.entities if e.label == "CHEMICAL"]

    @property
    def procedures(self) -> list[str]:
        return [e.text for e in self.entities if e.label == "PROCEDURE"]

    @property
    def anatomy(self) -> list[str]:
        return [e.text for e in self.entities if e.label == "ANATOMY"]

    def to_dict(self) -> dict:
        return {
            "note_id": self.note_id,
            "n_entities": len(self.entities),
            "diagnoses": self.diagnoses,
            "medications": self.medications,
            "procedures": self.procedures,
            "anatomy": self.anatomy,
            "entities_detail": [
                {"text": e.text, "label": e.label, "cui": e.cui, "confidence": e.confidence}
                for e in self.entities
            ],
        }


# ── Procedure Detection Patterns ───────────────────────────
_PROCEDURE_PATTERNS = [
    r"\b(colonoscopy|endoscopy|biopsy|excision|resection|arthroplasty|"
    r"appendectomy|cholecystectomy|hysterectomy|mastectomy|laparoscopy|"
    r"arthroscopy|bronchoscopy|catheterization|angioplasty|stent\s*placement|"
    r"intubation|tracheostomy|craniotomy|laminectomy|discectomy|"
    r"cesarean\s*section|c-section|angiography|echocardiogram|"
    r"electrocardiogram|ekg|ecg|mri|ct\s*scan|x-ray|xray|ultrasound|"
    r"mammography|pap\s*smear|blood\s*transfusion|dialysis|chemotherapy|"
    r"radiation\s*therapy|physical\s*therapy|suture|debridement|"
    r"amputation|transplant|bypass|grafting|drainage|aspiration|"
    r"injection|infusion|irrigation|cauterization|cryotherapy)\b",
]

_PROCEDURE_RE = re.compile("|".join(_PROCEDURE_PATTERNS), re.IGNORECASE)

# ── Anatomy Patterns ───────────────────────────────────────
_ANATOMY_PATTERNS = [
    r"\b(heart|lung|liver|kidney|brain|spleen|pancreas|stomach|"
    r"colon|rectum|bladder|uterus|ovary|prostate|thyroid|"
    r"adrenal|pituitary|esophagus|trachea|bronchus|aorta|"
    r"femur|tibia|humerus|spine|cervical|thoracic|lumbar|"
    r"sacral|pelvis|hip|knee|shoulder|elbow|wrist|ankle|"
    r"left\s*ventricle|right\s*ventricle|left\s*atrium|right\s*atrium)\b",
]

_ANATOMY_RE = re.compile("|".join(_ANATOMY_PATTERNS), re.IGNORECASE)


class NERExtractor:
    """Extract clinical entities from notes using scispaCy (with fallback).

    Parameters
    ----------
    spacy_model : str
        scispaCy model name (e.g. ``'en_core_sci_lg'``).
    ner_model : str
        scispaCy NER model name (e.g. ``'en_ner_bc5cdr_md'``).
    use_linker : bool
        If True, add the UMLS entity linker for CUI resolution.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_sci_lg",
        ner_model: str = "en_ner_bc5cdr_md",
        use_linker: bool = False,
    ) -> None:
        self.spacy_model_name = spacy_model
        self.ner_model_name = ner_model
        self.use_linker = use_linker
        self._nlp = None
        self._ner_nlp = None
        self._use_fallback = False

    def _load_models(self) -> None:
        """Lazy-load spaCy models."""
        if self._nlp is not None:
            return

        try:
            import spacy

            # Try loading the NER model first (it includes its own pipeline)
            try:
                self._ner_nlp = spacy.load(self.ner_model_name)
                logger.info("Loaded NER model: %s", self.ner_model_name)
            except Exception:
                logger.warning(
                    "NER model '%s' not found. Trying base model.",
                    self.ner_model_name,
                )

            # Load base model for general processing
            try:
                self._nlp = spacy.load(self.spacy_model_name)
                logger.info("Loaded base model: %s", self.spacy_model_name)
            except Exception:
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                    logger.warning("Fell back to en_core_web_sm")
                except Exception:
                    logger.warning("No spaCy model available — using regex fallback.")
                    self._use_fallback = True

            # Add UMLS linker if requested
            if self.use_linker and self._nlp and not self._use_fallback:
                try:
                    from scispacy.linking import EntityLinker  # noqa: F401

                    self._nlp.add_pipe(
                        "scispacy_linker",
                        config={"resolve_abbreviations": True, "linker_name": "umls"},
                    )
                    logger.info("UMLS entity linker added.")
                except Exception as e:
                    logger.warning("Could not add UMLS linker: %s", e)

        except ImportError:
            logger.warning("spaCy not installed — using regex fallback.")
            self._use_fallback = True

    def extract(self, text: str, note_id: str = "") -> NERResult:
        """Extract clinical entities from a single note.

        Parameters
        ----------
        text : str
            Clinical note text.
        note_id : str
            Identifier for the note.

        Returns
        -------
        NERResult
            Extracted entities.
        """
        self._load_models()
        preprocessed = preprocess_note(text)

        if self._use_fallback:
            return self._regex_extract(preprocessed, note_id)

        entities: list[ClinicalEntity] = []

        # Extract using NER model (diseases + chemicals)
        if self._ner_nlp:
            doc = self._ner_nlp(preprocessed[:100_000])  # Limit for safety
            for ent in doc.ents:
                entities.append(
                    ClinicalEntity(
                        text=ent.text.lower().strip(),
                        label=ent.label_,
                        start=ent.start_char,
                        end=ent.end_char,
                    )
                )

        # Extract procedures and anatomy with regex
        for match in _PROCEDURE_RE.finditer(preprocessed):
            entities.append(
                ClinicalEntity(
                    text=match.group().lower().strip(),
                    label="PROCEDURE",
                    start=match.start(),
                    end=match.end(),
                )
            )

        for match in _ANATOMY_RE.finditer(preprocessed):
            entities.append(
                ClinicalEntity(
                    text=match.group().lower().strip(),
                    label="ANATOMY",
                    start=match.start(),
                    end=match.end(),
                )
            )

        # Deduplicate
        seen = set()
        unique_entities = []
        for e in entities:
            key = (e.text, e.label)
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)

        return NERResult(note_id=note_id, entities=unique_entities)

    def _regex_extract(self, text: str, note_id: str) -> NERResult:
        """Fallback regex-based extraction when scispaCy is unavailable."""
        entities: list[ClinicalEntity] = []

        # Simple disease patterns
        disease_patterns = [
            r"\b(diabetes|hypertension|heart\s*failure|pneumonia|asthma|copd|"
            r"cancer|carcinoma|melanoma|lymphoma|leukemia|anemia|"
            r"arthritis|osteoporosis|epilepsy|migraine|depression|"
            r"anxiety|schizophrenia|alzheimer|parkinson|stroke|"
            r"myocardial\s*infarction|coronary\s*artery\s*disease|"
            r"atrial\s*fibrillation|chronic\s*kidney\s*disease|"
            r"gastroesophageal\s*reflux|hepatitis|cirrhosis|"
            r"pulmonary\s*embolism|deep\s*vein\s*thrombosis|"
            r"urinary\s*tract\s*infection|cellulitis|sepsis|"
            r"fracture|dislocation|sprain|contusion|laceration|"
            r"hypothyroidism|hyperthyroidism|obesity|malnutrition)\b",
        ]
        for pattern in disease_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    ClinicalEntity(
                        text=match.group().lower().strip(),
                        label="DISEASE",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.7,
                    )
                )

        # Medication patterns
        med_patterns = [
            r"\b(metformin|insulin|lisinopril|amlodipine|atorvastatin|"
            r"omeprazole|metoprolol|losartan|gabapentin|hydrochlorothiazide|"
            r"furosemide|warfarin|heparin|aspirin|ibuprofen|acetaminophen|"
            r"prednisone|levothyroxine|albuterol|amoxicillin|ciprofloxacin|"
            r"azithromycin|cephalexin|doxycycline|fluoxetine|sertraline|"
            r"morphine|oxycodone|hydrocodone|tramadol|diazepam|lorazepam)\b",
        ]
        for pattern in med_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    ClinicalEntity(
                        text=match.group().lower().strip(),
                        label="CHEMICAL",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.7,
                    )
                )

        # Procedures
        for match in _PROCEDURE_RE.finditer(text):
            entities.append(
                ClinicalEntity(
                    text=match.group().lower().strip(),
                    label="PROCEDURE",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                )
            )

        # Anatomy
        for match in _ANATOMY_RE.finditer(text):
            entities.append(
                ClinicalEntity(
                    text=match.group().lower().strip(),
                    label="ANATOMY",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                )
            )

        # Deduplicate
        seen = set()
        unique = []
        for e in entities:
            key = (e.text, e.label)
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return NERResult(note_id=note_id, entities=unique)

    def batch_extract(
        self,
        df: pd.DataFrame,
        text_col: str = "transcription",
        id_col: str = "note_id",
    ) -> list[NERResult]:
        """Extract entities from all notes in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with note text.
        text_col : str
            Text column name.
        id_col : str
            ID column name.

        Returns
        -------
        list[NERResult]
            One result per note.
        """
        results = []
        total = len(df)
        for i, (_, row) in enumerate(df.iterrows()):
            text = str(row.get(text_col, ""))
            nid = str(row.get(id_col, f"note_{i}"))
            result = self.extract(text, note_id=nid)
            results.append(result)

            if (i + 1) % 100 == 0:
                logger.info("NER progress: %d/%d (%.0f%%)", i + 1, total, 100 * (i + 1) / total)

        logger.info("NER extraction complete: %d notes processed", total)
        return results

    @staticmethod
    def results_to_dataframe(results: list[NERResult]) -> pd.DataFrame:
        """Convert NER results to a summary DataFrame.

        Parameters
        ----------
        results : list[NERResult]
            Extracted NER results.

        Returns
        -------
        pd.DataFrame
            One row per note with entity counts and lists.
        """
        rows = []
        for r in results:
            rows.append(
                {
                    "note_id": r.note_id,
                    "n_entities": len(r.entities),
                    "n_diagnoses": len(r.diagnoses),
                    "n_medications": len(r.medications),
                    "n_procedures": len(r.procedures),
                    "n_anatomy": len(r.anatomy),
                    "diagnoses": "; ".join(r.diagnoses),
                    "medications": "; ".join(r.medications),
                    "procedures": "; ".join(r.procedures),
                    "anatomy": "; ".join(r.anatomy),
                }
            )
        return pd.DataFrame(rows)
