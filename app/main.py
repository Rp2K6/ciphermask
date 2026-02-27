"""
FastAPI application entry point.

Wires up the service layer, mounts static files,
and exposes the ``POST /analyze`` and ``POST /analyze-file`` endpoints.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import spacy
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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

ALLOWED_EXTENSIONS = {".txt", ".csv", ".pdf", ".docx", ".png", ".jpg", ".jpeg"}


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


# -- Shared pipeline ---------------------------------------------------------

def _run_pipeline(text: str, include_sensitive: bool = True) -> AnalyzeResponse:
    raw_entities = pii_detector.detect(text)
    classified = classifier.classify(raw_entities)
    risk = risk_engine.compute_risk_score(classified)
    safety = risk_engine.compute_safety_score(risk)
    compliance = compliance_engine.evaluate(risk)
    cat_dist, label_dist = distribution_engine.compute_distributions(classified)

    redacted = None
    detected = None

    if include_sensitive:
        redacted = redaction_engine.redact(text, classified)
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


# -- Routes ------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Serve the SPA dashboard."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest):
    """Text-based privacy analysis pipeline."""
    return _run_pipeline(payload.text, payload.include_sensitive_output)


@app.post("/analyze-file", response_model=AnalyzeResponse)
async def analyze_file(
    file: UploadFile = File(...),
    include_sensitive_output: bool = Form(True),
):
    """File-based privacy analysis pipeline."""
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        text = pii_detector._load_text(tmp_path)

        return _run_pipeline(text, include_sensitive_output)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
