"""Source evidence: passages extracted from email/attachments, and citations."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.enums import ConflictStatus


class SourcePassage(BaseModel):
    """One page/segment of evidence extracted from an input document."""

    passage_id: str
    source_document_id: str
    source_type: str  # "email" | "attachment"
    page_number: int | None = None
    text: str
    ocr_applied: bool = False


class Citation(BaseModel):
    """Points a claimed fact back at the exact evidence it came from."""

    passage_id: str
    quoted_text: str
    page_number: int | None = None


class FieldValue(BaseModel):
    """A single extracted field: value plus everything a reviewer needs.

    Reviewer must see value, source, citation, confidence, and conflict
    status per the Case Workspace requirement in CLAUDE.md.
    """

    field_name: str
    value: str | None
    citations: list[Citation] = Field(default_factory=list)
    confidence: float
    conflict_status: ConflictStatus = ConflictStatus.NONE
    conflicting_values: list[str] = Field(default_factory=list)
