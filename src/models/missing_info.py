"""Missing/conflicting information gaps and the follow-up question they need."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MissingInformationItem(BaseModel):
    field_name: str
    reason: str  # "missing" | "conflicting"
    follow_up_question: str


class FollowUpDraft(BaseModel):
    draft_id: str
    case_id: str
    subject: str
    body: str
    editable: bool = True
    questions: list[MissingInformationItem] = Field(default_factory=list)
