"""Supervisor Agent: cross-cutting delegation/limits controller. Runs once
after planning and before the pipeline stages, checking retry/runtime/
token limits from procedural memory, persisting a checkpoint, and
confirming the route is still NORMAL before delegating to the first
pipeline agent. Downstream per-stage routing conditions (unsupported
file, low OCR quality, non-safety content, missing minimum criteria,
duplicate candidate) are evaluated by the graph's conditional edges after
each stage produces the evidence needed to decide them -- this node only
guards the case's overall resource limits and initial readiness."""

from __future__ import annotations

import time

from src.agents._common import call_tool, make_agent_event, make_tool_context
from src.memory.procedural import read_limits
from src.models.enums import AgentName, DecisionOutcome, RouteReason, ToolCallStatus
from src.models.reasoning import AgentDecision
from src.tools.workflow import (
    ReadCaseStateInput,
    RecordAuditEventInput,
    RouteCaseInput,
    WriteCaseCheckpointInput,
)

NODE = AgentName.SUPERVISOR


def run(state: dict) -> dict:
    started = time.monotonic()
    case_id = state["case_id"]
    trace_id = state["trace_id"]
    ctx = make_tool_context(case_id, trace_id, NODE)
    tool_events: list = []
    memory_events: list = []

    limits, limits_event = read_limits(ctx)
    memory_events.append(limits_event)
    max_agent_retries = limits.get("max_agent_retries", 1)

    retry_counts = state.get("retry_counts") or {}
    over_limit = any(count > max_agent_retries for count in retry_counts.values())

    call_tool("read_case_state", ReadCaseStateInput(case_id=case_id), ctx, tool_events)
    call_tool(
        "write_case_checkpoint",
        WriteCaseCheckpointInput(
            case_id=case_id,
            checkpoint_data={
                "current_step": "supervisor",
                "goal_status": state["goal_status"].value,
            },
        ),
        ctx,
        tool_events,
    )
    route_output = call_tool(
        "route_case",
        RouteCaseInput(llm_available=True, minimum_criteria_met=True),
        ctx,
        tool_events,
    )

    is_normal = route_output.route == RouteReason.NORMAL and not over_limit
    call_tool(
        "record_audit_event",
        RecordAuditEventInput(
            case_id=case_id,
            trace_id=trace_id,
            agent=NODE,
            action_name="supervisor_gate",
            status=ToolCallStatus.SUCCESS if is_normal else ToolCallStatus.ERROR,
            summary=f"route={route_output.route.value}; over_retry_limit={over_limit}; "
            f"max_agent_retries={max_agent_retries}.",
        ),
        ctx,
        tool_events,
    )

    decision = AgentDecision(
        agent=NODE,
        case_id=case_id,
        decision="delegate" if is_normal else "block",
        evidence=[f"route={route_output.route.value}", f"retry_counts={retry_counts}"],
        confidence=1.0,
        decision_summary=f"Supervisor gate: route={route_output.route.value}; "
        f"delegation {'approved' if is_normal else 'blocked (retry limit exceeded)'}.",
        next_action=DecisionOutcome.PROCEED if is_normal else DecisionOutcome.ESCALATE_TO_HUMAN,
        requires_human_review=not is_normal,
    )
    latency_ms = (time.monotonic() - started) * 1000
    agent_event = make_agent_event(
        decision,
        trace_id,
        node="supervisor",
        model_version=state["model_version"],
        latency_ms=latency_ms,
    )

    return {
        "current_step": "supervisor",
        "agent_events": [agent_event],
        "tool_events": tool_events,
        "memory_events": memory_events,
    }
