"""
Redaction engine.

Replaces every detected PII span with a bracketed label placeholder
while preserving the surrounding text structure.
"""

from __future__ import annotations


def redact(text: str, entities: list[dict]) -> str:
    """
    Return a copy of *text* with each entity replaced by ``[LABEL]``.

    Entities must carry ``start``, ``end``, and ``label`` keys.
    Replacement is performed back-to-front so that earlier offsets
    remain valid after each substitution.
    """
    # Work on a mutable list of characters to avoid repeated string concat
    sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)

    result = text
    for ent in sorted_entities:
        placeholder = f"[{ent['label']}]"
        result = result[: ent["start"]] + placeholder + result[ent["end"] :]

    return result
