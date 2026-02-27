"""
Classification service.

Maps each detected entity label to a privacy category
(Personal / Financial / Government) using the config mapping.
"""

from __future__ import annotations

from app.config import LABEL_CATEGORY_MAP

_FALLBACK_CATEGORY = "Personal"


def classify(entities: list[dict]) -> list[dict]:
    """
    Enrich each entity dict with a ``category`` key.

    Returns a *new* list (does not mutate the input).
    """
    classified: list[dict] = []
    for ent in entities:
        enriched = {**ent}
        enriched["category"] = LABEL_CATEGORY_MAP.get(
            ent["label"], _FALLBACK_CATEGORY
        )
        classified.append(enriched)
    return classified
