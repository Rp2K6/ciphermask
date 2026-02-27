"""
Configuration module for PII Redaction & Privacy Intelligence Platform.

Centralizes sensitivity weights, label-to-category mappings,
compliance thresholds, and spaCy model settings.
"""

# ---------------------------------------------------------------------------
# Sensitivity weights – higher weight = higher privacy risk
# ---------------------------------------------------------------------------
WEIGHTS: dict[str, int] = {
    "PHONE": 5,
    "EMAIL": 3,
    "PERSON": 2,
    "ORG": 1,
    "GPE": 1,
    "AADHAAR": 10,
    "PAN": 8,
    "CREDIT_CARD": 10,
    "IFSC": 6,
}

# ---------------------------------------------------------------------------
# Label → Category mapping
# ---------------------------------------------------------------------------
LABEL_CATEGORY_MAP: dict[str, str] = {
    "PERSON": "Personal",
    "PHONE": "Personal",
    "EMAIL": "Personal",
    "ORG": "Personal",
    "GPE": "Personal",
    "AADHAAR": "Government",
    "PAN": "Government",
    "CREDIT_CARD": "Financial",
    "IFSC": "Financial",
}

# ---------------------------------------------------------------------------
# Compliance thresholds (upper-bound exclusive)
# ---------------------------------------------------------------------------
COMPLIANCE_THRESHOLDS: list[tuple[int, str]] = [
    (80, "CRITICAL - NOT SAFE"),
    (50, "HIGH RISK - REVIEW REQUIRED"),
    (25, "MODERATE RISK - REVIEW"),
    (0, "SAFE FOR PUBLIC RELEASE"),
]

# ---------------------------------------------------------------------------
# spaCy model
# ---------------------------------------------------------------------------
SPACY_MODEL: str = "en_core_web_sm"

# Entity labels to extract from spaCy
SPACY_LABELS: set[str] = {"PERSON", "ORG", "GPE"}
