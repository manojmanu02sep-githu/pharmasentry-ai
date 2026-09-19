"""Seriousness triage: explicit indicators only, always human-confirmed.

The Seriousness Triage Agent detects EXPLICIT indicators present in the
evidence; it never infers or concludes seriousness on its own authority.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.enums import SeriousnessIndicator
from src.models.evidence import Citation


class SeriousnessFinding(BaseModel):
    indicator: SeriousnessIndicator
    citations: list[Citation] = Field(default_factory=list)


class SeriousnessTriageResult(BaseModel):
    findings: list[SeriousnessFinding] = Field(default_factory=list)
    requires_human_confirmation: bool = True

    @property
    def any_indicator_present(self) -> bool:
        return len(self.findings) > 0
