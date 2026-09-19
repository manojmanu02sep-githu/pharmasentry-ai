"""Triage: explicit indicators only, mapped to an AI-suggested priority.

The Triage Agent detects EXPLICIT indicators present in the evidence and
maps them, via a deterministic and documented rule, to a suggested
TriagePriority (LOW/MEDIUM/HIGH/CRITICAL). It never infers or concludes
priority on its own authority, and the suggestion is never final — human
review is mandatory before any action is taken (see CLAUDE.md Absolute
Boundaries).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.enums import TriageIndicator, TriagePriority
from src.models.evidence import Citation


class TriageFinding(BaseModel):
    indicator: TriageIndicator
    citations: list[Citation] = Field(default_factory=list)


class TriageResult(BaseModel):
    findings: list[TriageFinding] = Field(default_factory=list)
    suggested_priority: TriagePriority = TriagePriority.LOW
    rationale: str = ""
    requires_human_confirmation: bool = True

    @property
    def any_indicator_present(self) -> bool:
        return len(self.findings) > 0
