"""Duplicate candidates returned by hybrid retrieval. RAG never auto-merges."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DuplicateCandidate(BaseModel):
    candidate_case_id: str
    bm25_score: float
    vector_score: float
    combined_score: float
    matching_fields: list[str] = Field(default_factory=list)
    conflicting_fields: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
