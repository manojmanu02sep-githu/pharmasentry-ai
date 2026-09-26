"""Structured response models the LLM providers know how to fill.

Kept deliberately narrow (three shapes) so the mock provider can be
genuinely deterministic and auditable rather than a generic prompt parser.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SafetyClassificationResult(BaseModel):
    is_safety_report: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class NarrativeDraftResult(BaseModel):
    narrative_text: str


class QualityReviewResult(BaseModel):
    passed: bool
    feedback: str
