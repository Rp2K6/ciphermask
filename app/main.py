"""
FastAPI application entry point.

Wires up the service layer, mounts static files,
and exposes the ``POST /analyze`` endpoint.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import spacy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import SPACY_MODEL
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CategoryDistribution,
    DetectedEntity,
)
from app.services import (
    classifier,
    compliance_engine,
    distribution_engine,
    pii_detector,
    redaction_engine,
    risk_engine,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the spaCy model at startup so the first request is fast."""
    spacy.load(SPACY_MODEL)
    yield


app = FastAPI(
    title="PII Redaction & Privacy Intelligence Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# -- CORS (allow all origins for local / dev usage) -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Static files ------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -- Routes ------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Serve the SPA dashboard."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest):
    """
    Full privacy analysis pipeline:

    detect -> classify -> score -> comply -> redact -> distribute
    """
    text = payload.text

    # 1. Detection (hybrid: regex + NLP)
    raw_entities = pii_detector.detect(text)

    # 2. Classification
    classified = classifier.classify(raw_entities)

    # 3. Risk & safety scoring
    risk = risk_engine.compute_risk_score(classified)
    safety = risk_engine.compute_safety_score(risk)

    # 4. Compliance decision
    compliance = compliance_engine.evaluate(risk)

    # 5. Redaction
    redacted = redaction_engine.redact(text, classified)

    # 6. Distribution
    cat_dist, label_dist = distribution_engine.compute_distributions(classified)

    # 7. Build response entities (without internal offsets)
    detected = [
        DetectedEntity(
            value=e["value"],
            label=e["label"],
            category=e["category"],
            detection_method=e["detection_method"],
        )
        for e in classified
    ]

    return AnalyzeResponse(
        risk_score=risk,
        safety_score=safety,
        compliance_status=compliance,
        category_distribution=CategoryDistribution(**cat_dist),
        label_distribution=label_dist,
        redacted_text=redacted,
        detected_entities=detected,
    )
