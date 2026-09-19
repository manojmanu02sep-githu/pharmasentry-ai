"""Case-specific execution plan produced and revised by the Planner Agent."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from src.models.enums import AgentName


class PlanStep(BaseModel):
    step_id: str
    agent: AgentName
    description: str
    skip: bool = False
    skip_reason: str | None = None


class ExecutionPlan(BaseModel):
    plan_id: str
    case_id: str
    steps: list[PlanStep]
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PlanRevision(BaseModel):
    """One immutable entry in plan_history: what changed and why."""

    plan: ExecutionPlan
    reason: str
    revised_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
