"""Evaluator Agent output: quality checks over one step or the full trajectory."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationCheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class EvaluationResult(BaseModel):
    evaluation_id: str
    case_id: str
    step_name: str  # agent/node name, or "trajectory" for the end-to-end check
    checks: list[EvaluationCheck] = Field(default_factory=list)
    goal_completion: bool = False
    schema_valid: bool = True
    evidence_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    unresolved_conflicts: list[str] = Field(default_factory=list)
    requires_escalation: bool = False

    @property
    def passed(self) -> bool:
        return (
            all(c.passed for c in self.checks)
            and self.schema_valid
            and not self.unsupported_claims
        )
