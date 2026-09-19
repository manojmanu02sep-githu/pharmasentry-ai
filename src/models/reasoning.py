"""The single structured shape every bounded agent decision returns.

Reasoning capability requirement: each agent performs one bounded decision
over supplied evidence and returns exactly these fields. Raw chain-of-thought
is never part of this model and must never be persisted elsewhere.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from src.models.enums import AgentName, DecisionOutcome


class AgentDecision(BaseModel):
    agent: AgentName
    case_id: str
    decision: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    decision_summary: str
    next_action: DecisionOutcome
    requires_human_review: bool
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
