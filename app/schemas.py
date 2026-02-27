"""
Pydantic request / response schemas for the /analyze endpoint.
"""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=100_000,
        description="Raw text to analyse for PII.",
    )


class DetectedEntity(BaseModel):
    value: str
    label: str
    category: str
    detection_method: str  # "regex" | "nlp"


class CategoryDistribution(BaseModel):
    Personal: int = 0
    Financial: int = 0
    Government: int = 0


class AnalyzeResponse(BaseModel):
    risk_score: int
    safety_score: int
    compliance_status: str
    category_distribution: CategoryDistribution
    label_distribution: dict[str, int]
    redacted_text: str
    detected_entities: list[DetectedEntity]
