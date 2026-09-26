"""Conditional-route decision functions for the graph runner.

Each function is a cheap, pure recomputation over already-persisted,
truthful `CaseState` fields -- it never calls an LLM or a side-effecting
tool itself (the node that ran just before it already made the real,
audited tool calls and recorded its `decision_summary`/`agent_events`;
these functions only decide which node runs next). This mirrors the
reusable-pure-function pattern already used by
`src.agents.email_body_reader.classify_safety`.

Per CLAUDE.md's "Conditional routes": unsupported/unsafe files and
degraded OCR abandon further automated processing and go straight to
human review, since nothing downstream can be trusted to extract or
draft against unreliable evidence. Non-safety content likewise stops
early (closure confirmation, not a case package). All other conditions
(missing minimum criteria, potential duplicate, LLM unavailable) do not
require branching: they are already handled inline by the relevant node
(missing_information/follow_up_draft, duplicate_candidates, and the
deterministic LLM fallback in `call_llm_or_none`) and always continue to
Evaluator -> Human Review, which is mandatory for every case regardless.
"""

from __future__ import annotations

from src.agents.email_body_reader import classify_safety
from src.graph.state import CaseState
from src.models.enums import FileValidationStatus, OCRQuality

CONTINUE = "continue"
HUMAN_REVIEW = "human_review"


def after_subject_reader(state: CaseState) -> str:
    email = state.get("email")
    missing_fields = (
        [name for name in ("sender", "subject", "body") if not getattr(email, name, "").strip()]
        if email
        else ["email"]
    )
    invalid_attachments = [
        a
        for a in state.get("attachments") or []
        if a.validation_status != FileValidationStatus.VALID
    ]
    if missing_fields or invalid_attachments:
        return HUMAN_REVIEW
    return CONTINUE


def after_pdf_docx_reader(state: CaseState) -> str:
    degraded = [
        a
        for a in state.get("attachments") or []
        if a.ocr_quality in (OCRQuality.DEGRADED, OCRQuality.UNREADABLE)
    ]
    if degraded:
        return HUMAN_REVIEW
    return CONTINUE


def after_email_body_reader(state: CaseState) -> str:
    email = state.get("email")
    if email is None:
        return HUMAN_REVIEW
    classification = classify_safety(email.subject, email.body)
    if not classification.is_safety_report:
        return HUMAN_REVIEW
    return CONTINUE
