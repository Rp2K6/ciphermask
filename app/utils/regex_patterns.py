"""
Compiled regex patterns for structured PII detection.

Each pattern maps a canonical label to a compiled regex object.
Patterns are intentionally strict to minimise false positives.
"""

import re

PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "PHONE": re.compile(
        r"(?<!\d)"                       # no leading digit
        r"(?:\+?\d{1,3}[-.\s]?)?"        # optional country code
        r"(?:\(?\d{2,5}\)?[-.\s]?)?"     # optional area code
        r"\d{4,5}[-.\s]?\d{4,5}"         # core number
        r"(?!\d)"                        # no trailing digit
    ),
    "AADHAAR": re.compile(
        r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    ),
    "PAN": re.compile(
        r"\b[A-Z]{5}\d{4}[A-Z]\b"
    ),
    "CREDIT_CARD": re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    ),
    "IFSC": re.compile(
        r"\b[A-Z]{4}0[A-Z0-9]{6}\b"
    ),
}
