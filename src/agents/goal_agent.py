"""Goal Manager: creates the case's typed goal, success criteria, and
initial status. Bounded decision only -- does not evaluate whether
criteria are met (that's the Evaluator's job on each subsequent step)."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from src.agents._common import call_tool, make_agent_event, make_tool_context
from src.models.enums import AgentName, DecisionOutcome, GoalStatus, ToolCallStatus
from src.models.goal import CaseGoal, SuccessCriterion
from src.models.reasoning import AgentDecision
from src.tools.workflow import RecordAuditEventInput

NODE = AgentName.GOAL_AGENT

_SUCCESS_CRITERIA = (
    (
        "minimum_criteria_checked",
        "Minimum case criteria (patient, reporter, product, event) evaluated.",
    ),
    ("triage_evaluated", "Explicit seriousness indicators detected or ruled out."),
    ("duplicates_searched", "Synthetic case history searched for potential duplicates."),
    (
        "missing_information_identified",
        "Missing/conflicting information identified with follow-up questions.",
    ),
    ("narrative_drafted", "Source-cited case narrative draft prepared."),
    ("evaluated", "Case package evaluated for evidence coverage and unsupported claims."),
    ("human_review_reached", "Case paused for authorized human review."),
)


def run(state: dict) -> dict:
    started = time.monotonic()
    case_id = state["case_id"]
    trace_id = state["trace_id"]
    ctx = make_tool_context(case_id, trace_id, NODE)
    tool_events: list = []

    success_criteria = [
        SuccessCriterion(criterion_id=cid, description=desc, met=False)
        for cid, desc in _SUCCESS_CRITERIA
    ]
    goal = CaseGoal(
        case_id=case_id,
        success_criteria=success_criteria,
        status=GoalStatus.IN_PROGRESS,
        created_at=datetime.now(UTC),
    )

    call_tool(
        "record_audit_event",
        RecordAuditEventInput(
            case_id=case_id,
            trace_id=trace_id,
            agent=NODE,
            action_name="goal_created",
            status=ToolCallStatus.SUCCESS,
            summary=f"Case goal created with {len(success_criteria)} success criteria.",
        ),
        ctx,
        tool_events,
    )

    decision = AgentDecision(
        agent=NODE,
        case_id=case_id,
        decision="goal_created",
        evidence=[
            f"{len(success_criteria)} success criteria defined from "
            f"CLAUDE.md case package requirements."
        ],
        confidence=1.0,
        decision_summary=(
            f"Created case goal with {len(success_criteria)} success criteria; "
            f"boundaries enforced per CLAUDE.md."
        ),
        next_action=DecisionOutcome.PROCEED,
        requires_human_review=False,
    )
    latency_ms = (time.monotonic() - started) * 1000
    agent_event = make_agent_event(
        decision,
        trace_id,
        node="goal_agent",
        model_version=state["model_version"],
        latency_ms=latency_ms,
    )

    return {
        "goal": goal,
        "success_criteria": success_criteria,
        "goal_status": GoalStatus.IN_PROGRESS,
        "current_step": "goal_agent",
        "agent_events": [agent_event],
        "tool_events": tool_events,
    }
