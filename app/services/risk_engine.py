"""
Risk & safety scoring engine.

Computes a weighted risk score from detected entity counts
and derives a complementary safety score.
"""

from __future__ import annotations

from collections import Counter

from app.config import WEIGHTS


def compute_risk_score(entities: list[dict]) -> int:
    """Sum of (weight * count) for each label present."""
    counts = Counter(ent["label"] for ent in entities)
    score = sum(WEIGHTS.get(label, 0) * count for label, count in counts.items())
    return score


def compute_safety_score(risk_score: int) -> int:
    """Inverse of risk score, clamped to [0, 100]."""
    return max(0, 100 - risk_score)
