"""Typed case goal: what the case must achieve before the graph may stop."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from src.models.enums import GoalStatus


class SuccessCriterion(BaseModel):
    """One checkable condition. The goal is met only once every criterion is."""

    criterion_id: str
    description: str
    met: bool = False
    evidence_ref: str | None = None


class CaseGoal(BaseModel):
    """Created once by the Goal Manager agent; boundaries are non-negotiable."""

    case_id: str
    description: str = (
        "Produce a preliminary triage report for a diabetes-injection "
        "adverse-event email, with structured evidence-grounded fields, an "
        "AI-suggested triage priority (LOW/MEDIUM/HIGH/CRITICAL), and a "
        "completed human-review task. Never issue a final medical or "
        "regulatory decision."
    )
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    boundaries: list[str] = Field(
        default_factory=lambda: [
            "Never diagnose, recommend treatment, or infer causality.",
            "The AI triage priority (LOW/MEDIUM/HIGH/CRITICAL) is a "
            "suggestion only, never a final clinical or regulatory decision.",
            "Never make final expectedness, validity, reportability, or "
            "regulatory decisions.",
            "Never merge cases, send email, or submit to a regulator "
            "automatically.",
            "Every consequential output requires human approval.",
            "Never fabricate missing values, citations, metrics, or results.",
        ]
    )
    status: GoalStatus = GoalStatus.CREATED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def is_met(self) -> bool:
        return bool(self.success_criteria) and all(
            c.met for c in self.success_criteria
        )
