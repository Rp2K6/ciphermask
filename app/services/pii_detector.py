"""
Hybrid PII detection service.

Combines regex-based pattern matching (structured identifiers) with
spaCy NLP (contextual entities) to produce a unified list of detected
PII entities.
"""

from __future__ import annotations

import spacy
from spacy.language import Language

from app.config import SPACY_MODEL, SPACY_LABELS
from app.utils.regex_patterns import PATTERNS


_nlp: Language | None = None


def _get_nlp() -> Language:
    """Lazy-load the spaCy model once."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(SPACY_MODEL)
    return _nlp


def _detect_regex(text: str) -> list[dict]:
    """Run every compiled regex pattern against *text*."""
    entities: list[dict] = []
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append(
                {
                    "value": match.group(),
                    "label": label,
                    "detection_method": "regex",
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return entities


def _detect_nlp(text: str) -> list[dict]:
    """Run spaCy NER and return entities whose label is in SPACY_LABELS."""
    nlp = _get_nlp()
    doc = nlp(text)
    entities: list[dict] = []
    for ent in doc.ents:
        if ent.label_ in SPACY_LABELS:
            entities.append(
                {
                    "value": ent.text,
                    "label": ent.label_,
                    "detection_method": "nlp",
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
            )
    return entities


def _deduplicate(entities: list[dict]) -> list[dict]:
    """Remove overlapping spans, preferring regex matches (more precise)."""
    # Sort: regex first, then by start position
    priority = {"regex": 0, "nlp": 1}
    entities.sort(key=lambda e: (priority.get(e["detection_method"], 2), e["start"]))

    kept: list[dict] = []
    occupied: list[tuple[int, int]] = []

    for ent in entities:
        s, e = ent["start"], ent["end"]
        if any(s < oe and e > os for os, oe in occupied):
            continue  # overlaps an already-kept span
        kept.append(ent)
        occupied.append((s, e))

    # Return in document order
    kept.sort(key=lambda e: e["start"])
    return kept


def detect(text: str) -> list[dict]:
    """
    Run hybrid detection and return deduplicated entity list.

    Each entity dict contains:
        value, label, detection_method, start, end
    """
    regex_entities = _detect_regex(text)
    nlp_entities = _detect_nlp(text)
    merged = regex_entities + nlp_entities
    return _deduplicate(merged)
