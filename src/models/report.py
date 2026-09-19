"""Chronological, source-cited triage report built from validated facts only.

Produced by the Report Generator Agent, which replaces the prior Narrative
Agent and absorbs missing-information/follow-up drafting for this workflow
(see CLAUDE.md "Agents" and the Phase 7 notes in progress.md).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.evidence import Citation


class ReportSection(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)


class TriageReport(BaseModel):
    report_id: str
    case_id: str
    sections: list[ReportSection] = Field(default_factory=list)
    unsupported_claim_flags: list[str] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.sections)
