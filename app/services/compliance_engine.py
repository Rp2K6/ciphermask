"""
Compliance decision engine.

Evaluates the risk score against configurable thresholds
and returns a human-readable compliance status string.
"""

from __future__ import annotations

from app.config import COMPLIANCE_THRESHOLDS


def evaluate(risk_score: int) -> str:
    """Return the compliance status label for the given *risk_score*."""
    for threshold, status in COMPLIANCE_THRESHOLDS:
        if risk_score > threshold:
            return status
    # Fallback (should not be reached with properly ordered thresholds)
    return COMPLIANCE_THRESHOLDS[-1][1]
