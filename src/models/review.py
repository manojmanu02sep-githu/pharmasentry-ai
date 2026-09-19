"""Human review: the only path that can approve, reject, or edit a case."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from src.models.enums import ReviewDecision


class ReviewerChange(BaseModel):
    field_name: str
    old_value: str | None
    new_value: str | None
    reviewer_id: str
    changed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reason: str | None = None


class ReviewStatus(BaseModel):
    case_id: str
    decision: ReviewDecision = ReviewDecision.PENDING
    reviewer_id: str | None = None
    notes: str | None = None
    decided_at: datetime | None = None
