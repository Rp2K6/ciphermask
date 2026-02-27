from __future__ import annotations

import os
import re
from typing import Dict, List

import pandas as pd
import spacy
from spacy.language import Language

from app.config import SPACY_MODEL, SPACY_LABELS

_nlp: Language | None = None

_REGEX_PATTERNS: List[tuple] = [
    ("CREDIT_CARD", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")),
    ("AADHAAR", re.compile(r"(?<!\+)(?<!\d)\b\d{4}\s?\d{4}\s?\d{4}\b(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+91[-\s]?)?[6-9]\d{4}[-\s]?\d{5}(?!\d)")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
]

_NAME_INTRO = re.compile(
    r"\b(?:my\s+name\s+is|i\s+am|this\s+is)\s+",
    re.IGNORECASE,
)

_GREETING_INTRO = re.compile(
    r"\b(?:hi|hello|hey)\s+",
    re.IGNORECASE,
)

_FIELD_LABEL = re.compile(
    r"(?:^|[\n,;])[ \t]*(?:(?:Full\s+)?Name|Emergency\s+Contact|Contact\s+Person|Contact\s+Name|Patient\s+Name|Employee\s+Name|Student\s+Name|Applicant\s+Name|Customer\s+Name|Guardian\s+Name|Father(?:'?s)?\s+Name|Mother(?:'?s)?\s+Name|Spouse(?:'?s)?\s+Name)[ \t]*[:]\s*",
    re.IGNORECASE | re.MULTILINE,
)

_NAME_WORD_RE = re.compile(r"[A-Za-z]{2,20}")

_STRUCTURAL_KEYWORDS = frozenset({
    "credit card", "credit", "card",
    "emergency contact", "emergency",
    "phone", "phone number", "mobile", "contact",
    "name", "full name", "first name", "last name",
    "email", "email address", "e-mail",
    "aadhaar", "aadhar", "aadhaar number",
    "pan", "pan number", "pan card",
    "address", "dob", "date of birth",
    "gender", "age", "occupation",
    "father", "mother", "spouse", "guardian",
    "ifsc", "account", "bank",
})

_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet",
    "in", "on", "at", "to", "for", "of", "with", "from", "by", "as",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "i", "me", "my", "mine", "we", "us", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs",
    "this", "that", "these", "those",
    "who", "whom", "whose", "which", "what", "where", "when", "how", "why",
    "not", "no", "if", "then", "else", "than",
    "here", "there", "now", "just", "also", "very", "really",
    "everyone", "everybody", "guys", "team", "sir", "madam",
    "dear", "friend", "friends", "folks", "mate", "world",
    "fine", "good", "great", "ok", "okay", "sorry", "happy",
    "glad", "sure", "ready", "done", "well", "all", "back",
    "going", "looking", "trying", "working", "writing", "calling",
    "sending", "using", "new", "old", "need", "want", "know",
    "about", "into", "over", "after", "before", "between",
    "through", "during", "without", "under", "above",
})

_BUSINESS_SUFFIXES = frozenset({
    "ltd", "pvt", "inc", "corp", "technologies", "solutions",
    "systems", "private", "limited", "llc", "llp", "co", "company",
    "services", "enterprises", "group", "industries", "consulting",
    "foundation", "institute", "association", "corporation",
})

_FILE_EXTENSIONS = {".txt", ".csv", ".pdf", ".docx", ".png", ".jpg", ".jpeg"}


def _get_nlp() -> Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(SPACY_MODEL)
    return _nlp


def _is_structural(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _STRUCTURAL_KEYWORDS:
        return True
    for kw in _STRUCTURAL_KEYWORDS:
        if len(kw.split()) > 1 and kw in normalized:
            return True
    return False


def _is_file_path(input_data: str) -> bool:
    stripped = input_data.strip()
    if len(stripped) > 500 or "\n" in stripped:
        return False
    _, ext = os.path.splitext(stripped.lower())
    return ext in _FILE_EXTENSIONS and os.path.isfile(stripped)


def _load_text(input_data: str) -> str:
    if not isinstance(input_data, str):
        return str(input_data)

    stripped = input_data.strip()
    if not _is_file_path(stripped):
        return input_data

    ext = os.path.splitext(stripped.lower())[1]

    if ext == ".txt":
        with open(stripped, "r", encoding="utf-8") as f:
            return f.read()

    if ext == ".csv":
        df = pd.read_csv(stripped, dtype=str).fillna("")
        return " ".join(df.astype(str).values.flatten())

    if ext == ".pdf":
        import pdfplumber
        pages: List[str] = []
        with pdfplumber.open(stripped) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)

    if ext == ".docx":
        from docx import Document
        doc = Document(stripped)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    if ext in {".png", ".jpg", ".jpeg"}:
        from PIL import Image
        import pytesseract
        img = Image.open(stripped)
        return pytesseract.image_to_string(img)

    return input_data


def _detect_regex(text: str) -> List[Dict]:
    entities: List[Dict] = []
    for label, pattern in _REGEX_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group()
            digits = re.sub(r"\D", "", value)
            if label == "CREDIT_CARD" and len(digits) != 16:
                continue
            if label == "AADHAAR" and len(digits) != 12:
                continue
            if label == "PHONE":
                if digits.startswith("91") and len(digits) == 12:
                    digits = digits[2:]
                if len(digits) != 10:
                    continue
            entities.append({
                "value": value,
                "label": label,
                "detection_method": "regex",
                "start": match.start(),
                "end": match.end(),
            })
    return entities


def _detect_nlp(text: str) -> List[Dict]:
    nlp = _get_nlp()
    doc = nlp(text)
    entities: List[Dict] = []
    for ent in doc.ents:
        if ent.label_ not in SPACY_LABELS:
            continue

        tokens = list(ent)
        if ent.label_ in ("PERSON", "ORG"):
            raw_value = ent.text.strip()
            if _is_structural(raw_value):
                continue

        if ent.label_ == "PERSON":
            clean = []
            for t in tokens:
                word = t.text.strip()
                if not word:
                    continue
                if word.lower() in _STOP_WORDS:
                    break
                bare = word.replace("'", "").replace("-", "")
                if not bare.isalpha() or len(bare) < 2:
                    break
                clean.append(t)
            if not clean or len(clean) > 4:
                continue
            start = clean[0].idx
            end = clean[-1].idx + len(clean[-1])
            value = text[start:end]
            if _is_structural(value):
                continue
        else:
            value = ent.text.strip()
            start = ent.start_char
            end = ent.end_char

        entities.append({
            "value": value,
            "label": ent.label_,
            "detection_method": "nlp",
            "start": start,
            "end": end,
        })
    return entities


def _extract_names_from_pattern(
    text: str,
    pattern: re.Pattern,
    max_words: int,
) -> List[Dict]:
    entities: List[Dict] = []
    for intro in pattern.finditer(text):
        words: list = []
        search_pos = intro.end()
        for m in _NAME_WORD_RE.finditer(text, search_pos):
            if m.start() != search_pos:
                gap = text[search_pos:m.start()]
                if not all(c == " " for c in gap):
                    break
            if m.group().lower() in _STOP_WORDS:
                break
            if _is_structural(m.group()):
                break
            words.append((m.group(), m.start(), m.end()))
            search_pos = m.end()
            if len(words) >= max_words:
                break
        if words:
            name_start = words[0][1]
            name_end = words[-1][2]
            candidate = text[name_start:name_end]
            if not _is_structural(candidate):
                entities.append({
                    "value": candidate,
                    "label": "PERSON",
                    "detection_method": "rule",
                    "start": name_start,
                    "end": name_end,
                })
    return entities


def _apply_name_rules(text: str) -> List[Dict]:
    entities: List[Dict] = []
    entities.extend(_extract_names_from_pattern(text, _NAME_INTRO, 4))
    entities.extend(_extract_names_from_pattern(text, _GREETING_INTRO, 2))
    entities.extend(_extract_names_from_pattern(text, _FIELD_LABEL, 4))
    return entities


def _resolve_overlaps(entities: List[Dict]) -> List[Dict]:
    if not entities:
        return []

    label_priority = {label: i for i, (label, _) in enumerate(_REGEX_PATTERNS)}
    method_priority = {"regex": 0, "rule": 1, "nlp": 2}

    def sort_key(e: Dict) -> tuple:
        mp = method_priority.get(e["detection_method"], 99)
        lp = label_priority.get(e["label"], 99)
        return (mp, lp)

    entities.sort(key=lambda e: (sort_key(e), e["start"]))

    kept: List[Dict] = []
    occupied: List[tuple] = []
    for ent in entities:
        s, e = ent["start"], ent["end"]
        if any(s < oe and e > os for os, oe in occupied):
            continue
        kept.append(ent)
        occupied.append((s, e))

    seen: set = set()
    deduped: List[Dict] = []
    for ent in kept:
        key = (ent["start"], ent["end"], ent["label"])
        if key not in seen:
            seen.add(key)
            deduped.append(ent)

    deduped.sort(key=lambda e: e["start"])
    return deduped


def detect(input_data: str) -> List[Dict]:
    text = _load_text(input_data)

    regex_entities = _detect_regex(text)
    nlp_entities = _detect_nlp(text)

    for ent in nlp_entities:
        if ent["label"] == "ORG":
            if _is_structural(ent["value"]):
                ent["_discard"] = True
                continue
            w = ent["value"].split()
            if (1 <= len(w) <= 3
                    and all(tok.isalpha() for tok in w)
                    and not any(tok.lower() in _BUSINESS_SUFFIXES for tok in w)):
                ent["label"] = "PERSON"

    nlp_entities = [e for e in nlp_entities if not e.get("_discard")]
    for ent in nlp_entities:
        ent.pop("_discard", None)

    rule_entities = _apply_name_rules(text)
    merged = regex_entities + nlp_entities + rule_entities
    return _resolve_overlaps(merged)
