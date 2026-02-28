"""
Image redaction service.

Uses Tesseract OCR to extract text with bounding-box coordinates,
runs PII detection via the existing pipeline, and draws filled
black rectangles over every detected PII region in the image.
"""

from __future__ import annotations

import io

import pytesseract
from PIL import Image, ImageDraw

from app.services import classifier, pii_detector

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _draw_black_rect(
    draw: ImageDraw.ImageDraw, w: dict, padding: int
) -> None:
    draw.rectangle(
        [
            w["left"] - padding,
            w["top"] - padding,
            w["left"] + w["width"] + padding,
            w["top"] + w["height"] + padding,
        ],
        fill="black",
    )


def _redact_by_text_match(
    draw: ImageDraw.ImageDraw,
    words: list[dict],
    entity_value: str,
    padding: int,
) -> None:
    """Fallback: find and redact OCR words whose text matches *entity_value*."""
    entity_tokens = entity_value.split()
    if not entity_tokens:
        return

    n = len(entity_tokens)
    for i in range(len(words) - n + 1):
        if all(
            words[i + j]["text"].lower() == entity_tokens[j].lower()
            for j in range(n)
        ):
            for j in range(n):
                _draw_black_rect(draw, words[i + j], padding)


def redact_image(image_path: str) -> bytes:
    """Return PNG bytes of *image_path* with PII regions blacked out."""
    img = Image.open(image_path).convert("RGB")

    # OCR with word-level bounding boxes
    ocr = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    # Build full text preserving line structure from OCR layout so that
    # spaCy NER and the rule-based field-label patterns work correctly.
    words: list[dict] = []
    parts: list[str] = []
    offset = 0
    prev_block = prev_par = prev_line = -1

    for i in range(len(ocr["text"])):
        word = ocr["text"][i].strip()
        if not word:
            continue

        cur_block = ocr["block_num"][i]
        cur_par = ocr["par_num"][i]
        cur_line = ocr["line_num"][i]

        if parts:
            # Newline between different OCR lines/blocks; space within a line.
            if (
                cur_block != prev_block
                or cur_par != prev_par
                or cur_line != prev_line
            ):
                parts.append("\n")
            else:
                parts.append(" ")
            offset += 1

        words.append(
            {
                "text": word,
                "char_start": offset,
                "char_end": offset + len(word),
                "left": ocr["left"][i],
                "top": ocr["top"][i],
                "width": ocr["width"][i],
                "height": ocr["height"][i],
            }
        )
        parts.append(word)
        offset += len(word)

        prev_block, prev_par, prev_line = cur_block, cur_par, cur_line

    full_text = "".join(parts)

    if not full_text.strip():
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # Detect & classify PII using the existing pipeline
    raw_entities = pii_detector.detect(full_text)
    classified = classifier.classify(raw_entities)

    if not classified:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # Draw black rectangles over every word that overlaps a PII entity
    draw = ImageDraw.Draw(img)
    padding = 3

    for entity in classified:
        ent_start = entity["start"]
        ent_end = entity["end"]

        # Primary: character-offset overlap
        matched = False
        for w in words:
            if w["char_end"] > ent_start and w["char_start"] < ent_end:
                _draw_black_rect(draw, w, padding)
                matched = True

        # Fallback: direct text matching when offsets find nothing
        if not matched:
            _redact_by_text_match(draw, words, entity["value"], padding)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
