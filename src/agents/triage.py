"""Seriousness Triage Agent: detects explicit seriousness-indicator
phrases only (no inference) and always requires human confirmation --
`suggested_priority` is AI-suggested, never a final seriousness decision."""

from __future__ import annotations

import time

from src.agents._common import call_tool, find_citations, make_agent_event, make_tool_context
from src.models.enums import AgentName, DecisionOutcome, ToolCallStatus
from src.models.reasoning import AgentDecision
from src.models.triage import TriageFinding, TriageResult
from src.tools.domain import DetectExplicitTriageIndicatorsInput, SuggestTriagePriorityInput
from src.tools.workflow import RecordAuditEventInput

NODE = AgentName.TRIAGE


def run(state: dict) -> dict:
    started = time.monotonic()
    case_id = state["case_id"]
    trace_id = state["trace_id"]
    ctx = make_tool_context(case_id, trace_id, NODE)
    tool_events: list = []

    passages = state.get("source_passages") or []
    combined_text = "\n".join(p.text for p in passages)

    detect_output = call_tool(
        "detect_explicit_triage_indicators",
        DetectExplicitTriageIndicatorsInput(text=combined_text),
        ctx,
        tool_events,
    )
    indicators = detect_output.indicators
    priority_output = call_tool(
        "suggest_triage_priority",
        SuggestTriagePriorityInput(indicators=indicators),
        ctx,
        tool_events,
    )

    findings = [
        TriageFinding(
            indicator=match.indicator,
            citations=find_citations(passages, match.matched_phrase),
        )
        for match in detect_output.matches
    ]
    triage_result = TriageResult(
        findings=findings,
        suggested_priority=priority_output.suggested_priority,
        rationale=priority_output.rationale,
        requires_human_confirmation=True,
    )

    call_tool(
        "record_audit_event",
        RecordAuditEventInput(
            case_id=case_id,
            trace_id=trace_id,
            agent=NODE,
            action_name="triage_assessed",
            status=ToolCallStatus.SUCCESS,
            summary=(
                f"{len(findings)} explicit indicator(s) found; "
                f"suggested_priority={priority_output.suggested_priority.value}."
            ),
        ),
        ctx,
        tool_events,
    )

    decision = AgentDecision(
        agent=NODE,
        case_id=case_id,
        decision="indicators_detected" if findings else "no_explicit_indicators",
        evidence=[m.matched_phrase for m in detect_output.matches],
        confidence=0.9 if findings else 0.6,
        decision_summary=(
            f"Detected {len(findings)} explicit seriousness indicator(s); "
            f"AI-suggested priority={priority_output.suggested_priority.value} "
            f"(requires human confirmation)."
        ),
        next_action=DecisionOutcome.ESCALATE_TO_HUMAN if findings else DecisionOutcome.PROCEED,
        requires_human_review=True,
    )
    latency_ms = (time.monotonic() - started) * 1000
    agent_event = make_agent_event(
        decision,
        trace_id,
        node="triage",
        model_version=state["model_version"],
        latency_ms=latency_ms,
    )

    return {
        "triage_result": triage_result,
        "current_step": "triage",
        "agent_events": [agent_event],
        "tool_events": tool_events,
    }
