"""Email Body Reader: second Intake-stage node. Classifies the body for
safety relevance (via the LLM abstraction -- classification is an LLM
concern per CLAUDE.md, not a deterministic tool) and enforces size limits
on the body content before it is normalized into source passages."""

from __future__ import annotations

import time

from src.agents._common import call_llm_or_none, call_tool, make_agent_event, make_tool_context
from src.llm.schemas import SafetyClassificationResult
from src.models.enums import AgentName, DecisionOutcome, ToolCallStatus
from src.models.reasoning import AgentDecision
from src.tools.intake import EnforceFileSizeInput, SanitizeFilenameInput
from src.tools.workflow import RecordAuditEventInput

NODE = AgentName.EMAIL_BODY_READER


def classify_safety(subject: str, body: str) -> SafetyClassificationResult:
    """Deterministic mock-LLM safety classification, callable standalone
    so the graph's routing function can recompute the same result without
    depending on a state field the CaseState schema doesn't carry."""
    result, _ = call_llm_or_none(
        SafetyClassificationResult,
        context={"email_subject": subject, "email_body": body},
    )
    if result is None:
        # LLM unavailable: deterministic fallback per CLAUDE.md -- treat as
        # requiring human review rather than guessing.
        return SafetyClassificationResult(
            is_safety_report=False,
            confidence=0.0,
            rationale="LLM unavailable; deterministic fallback.",
        )
    return result


def run(state: dict) -> dict:
    started = time.monotonic()
    case_id = state["case_id"]
    trace_id = state["trace_id"]
    ctx = make_tool_context(case_id, trace_id, NODE)
    tool_events: list = []

    email = state["email"]
    body_bytes = email.body.encode("utf-8")

    size_output = call_tool(
        "enforce_file_size", EnforceFileSizeInput(size_bytes=len(body_bytes)), ctx, tool_events
    )
    sanitize_output = call_tool(
        "sanitize_filename",
        SanitizeFilenameInput(filename=f"{email.subject}.txt"),
        ctx,
        tool_events,
    )
    classification = classify_safety(email.subject, email.body)

    call_tool(
        "record_audit_event",
        RecordAuditEventInput(
            case_id=case_id,
            trace_id=trace_id,
            agent=NODE,
            action_name="body_classified",
            status=ToolCallStatus.SUCCESS,
            summary=f"is_safety_report={classification.is_safety_report} "
            f"confidence={classification.confidence:.2f}; "
            f"body_size_ok={size_output.within_limit}.",
        ),
        ctx,
        tool_events,
    )

    next_action = (
        DecisionOutcome.PROCEED
        if classification.is_safety_report
        else DecisionOutcome.ESCALATE_TO_HUMAN
    )
    content_label = (
        "a possible safety report" if classification.is_safety_report else "non-safety content"
    )
    decision = AgentDecision(
        agent=NODE,
        case_id=case_id,
        decision="safety_report" if classification.is_safety_report else "non_safety_content",
        evidence=[classification.rationale],
        confidence=classification.confidence,
        decision_summary=f"Classified email body as {content_label} "
        f"(confidence={classification.confidence:.2f}); "
        f"sanitized filename={sanitize_output.sanitized_filename!r}.",
        next_action=next_action,
        requires_human_review=not classification.is_safety_report,
    )
    latency_ms = (time.monotonic() - started) * 1000
    agent_event = make_agent_event(
        decision,
        trace_id,
        node="email_body_reader",
        model_version=state["model_version"],
        latency_ms=latency_ms,
    )

    return {
        "current_step": "email_body_reader",
        "agent_events": [agent_event],
        "tool_events": tool_events,
    }
