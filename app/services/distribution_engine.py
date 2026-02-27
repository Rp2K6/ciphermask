"""
Distribution engine.

Aggregates detected entities into category-level and label-level
frequency distributions for dashboard visualisation.
"""

from __future__ import annotations

from collections import Counter


def compute_distributions(
    entities: list[dict],
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Return ``(category_distribution, label_distribution)`` dicts.

    Both are simple ``{name: count}`` mappings.
    """
    category_dist: dict[str, int] = Counter()
    label_dist: dict[str, int] = Counter()

    for ent in entities:
        category_dist[ent["category"]] += 1
        label_dist[ent["label"]] += 1

    return dict(category_dist), dict(label_dist)
