"""Human review: the only path that can approve, reject, or edit a case."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from src.models.enums import ReviewDecision, TriagePriority


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


class ReviewTask(BaseModel):
    """One queued item in the Human Review Queue — created before any
    reviewer decision exists (that's `ReviewStatus`, above, once decided)."""

    task_id: str
    case_id: str
    reason: str
    evidence_summary: str
    ai_suggestion: str | None = None
    ai_suggested_priority: TriagePriority | None = None
    status: Literal["pending", "in_review", "resolved"] = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
