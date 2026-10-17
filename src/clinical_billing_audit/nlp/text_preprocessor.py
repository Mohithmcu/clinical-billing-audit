"""Clinical text preprocessor.

Normalises clinical notes before NER and feature extraction:
- Section header detection and segmentation
- Abbreviation expansion
- De-identification artefact cleanup
- Sentence splitting optimised for clinical text
"""

from __future__ import annotations

import re
from typing import Iterator

from clinical_billing_audit.utils.constants import CLINICAL_ABBREVIATIONS
from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)

# ── Section Header Patterns ─────────────────────────────────
_SECTION_HEADERS = [
    "CHIEF COMPLAINT",
    "HISTORY OF PRESENT ILLNESS",
    "PAST MEDICAL HISTORY",
    "MEDICATIONS",
    "ALLERGIES",
    "REVIEW OF SYSTEMS",
    "PHYSICAL EXAMINATION",
    "ASSESSMENT AND PLAN",
    "ASSESSMENT",
    "PLAN",
    "DIAGNOSES",
    "DIAGNOSIS",
    "PROCEDURE",
    "PROCEDURES",
    "OPERATIVE FINDINGS",
    "PREOPERATIVE DIAGNOSIS",
    "POSTOPERATIVE DIAGNOSIS",
    "INDICATION",
    "INDICATIONS",
    "SOCIAL HISTORY",
    "FAMILY HISTORY",
    "LABORATORY DATA",
    "LAB DATA",
    "IMAGING",
    "IMPRESSION",
    "RECOMMENDATIONS",
    "DISPOSITION",
    "DISCHARGE MEDICATIONS",
    "DISCHARGE INSTRUCTIONS",
    "VITAL SIGNS",
]

_SECTION_RE = re.compile(
    r"^(" + "|".join(re.escape(h) for h in _SECTION_HEADERS) + r")\s*[:\-]?\s*",
    re.IGNORECASE | re.MULTILINE,
)

# ── Cleanup Patterns ────────────────────────────────────────
_PATTERNS = {
    "phi_brackets": re.compile(r"\[\*\*[^\]]*\*\*\]"),
    "multiple_spaces": re.compile(r"[ \t]{2,}"),
    "multiple_newlines": re.compile(r"\n{3,}"),
    "numbered_list": re.compile(r"^\s*\d+[\.\)]\s*", re.MULTILINE),
    "bullet_list": re.compile(r"^\s*[-\*•]\s*", re.MULTILINE),
}


def preprocess_note(
    text: str,
    expand_abbreviations: bool = True,
    remove_phi_placeholders: bool = True,
    lowercase: bool = False,
) -> str:
    """Clean and normalise a single clinical note.

    Parameters
    ----------
    text : str
        Raw clinical note text.
    expand_abbreviations : bool
        If True, expand known clinical abbreviations.
    remove_phi_placeholders : bool
        If True, remove ``[**...**]``-style PHI markers.
    lowercase : bool
        If True, convert to lowercase (usually False for NER).

    Returns
    -------
    str
        Preprocessed note text.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove PHI placeholders
    if remove_phi_placeholders:
        text = _PATTERNS["phi_brackets"].sub("", text)

    # Normalise whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _PATTERNS["multiple_spaces"].sub(" ", text)
    text = _PATTERNS["multiple_newlines"].sub("\n\n", text)

    # Expand abbreviations (word-boundary aware)
    if expand_abbreviations:
        for abbr, expansion in CLINICAL_ABBREVIATIONS.items():
            pattern = re.compile(r"\b" + re.escape(abbr) + r"\b", re.IGNORECASE)
            text = pattern.sub(expansion, text)

    # Optional lowercase
    if lowercase:
        text = text.lower()

    return text.strip()


def segment_sections(text: str) -> dict[str, str]:
    """Split a clinical note into named sections.

    Parameters
    ----------
    text : str
        Clinical note text (ideally preprocessed).

    Returns
    -------
    dict[str, str]
        Mapping from section header → section body text.
        A ``'__preamble__'`` key captures text before the first header.
    """
    sections: dict[str, str] = {}
    current_header = "__preamble__"
    current_lines: list[str] = []

    for line in text.split("\n"):
        match = _SECTION_RE.match(line.strip())
        if match:
            # Save previous section
            sections[current_header] = "\n".join(current_lines).strip()
            current_header = match.group(1).strip().upper()
            remainder = line[match.end():].strip()
            current_lines = [remainder] if remainder else []
        else:
            current_lines.append(line)

    # Save last section
    sections[current_header] = "\n".join(current_lines).strip()

    # Remove empty sections
    sections = {k: v for k, v in sections.items() if v}
    return sections


def extract_sentences(text: str) -> list[str]:
    """Split clinical text into sentences (handles abbreviation-heavy text).

    Parameters
    ----------
    text : str
        Clinical note text.

    Returns
    -------
    list[str]
        List of sentences.
    """
    # Protect common abbreviations from splitting
    protected = text
    for abbr in ["Dr.", "Mr.", "Mrs.", "Ms.", "vs.", "etc.", "i.e.", "e.g.", "pt.", "q.d."]:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)

    # Restore dots
    sentences = [s.replace("<DOT>", ".").strip() for s in sentences]
    return [s for s in sentences if len(s) > 5]  # Drop very short fragments


def batch_preprocess(
    texts: Iterator[str] | list[str],
    **kwargs,
) -> list[str]:
    """Preprocess a batch of clinical notes.

    Parameters
    ----------
    texts : iterable of str
        Raw note texts.
    **kwargs
        Forwarded to :func:`preprocess_note`.

    Returns
    -------
    list[str]
        Preprocessed texts.
    """
    return [preprocess_note(t, **kwargs) for t in texts]
