"""Chronological, source-cited narrative draft built from validated facts only."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.evidence import Citation


class NarrativeSentence(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)


class NarrativeDraft(BaseModel):
    draft_id: str
    case_id: str
    sentences: list[NarrativeSentence] = Field(default_factory=list)
    unsupported_claim_flags: list[str] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.sentences)
