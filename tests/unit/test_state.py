"""Unit tests for the typed LangGraph CaseState (src/graph/state.py)."""

from __future__ import annotations

import operator

from src.graph.state import create_initial_state, new_case_id, new_trace_id
from src.models import EmailMessage, GoalStatus
from src.models.audit import AgentEvent
from src.models.enums import AgentName


def _demo_email() -> EmailMessage:
    return EmailMessage(
        message_id="demo-case-001",
        sender="r.tanaka@fictionalclinic-demo.example",
        subject="Possible adverse event report - DemoGluca patient hospitalized",
        body="Synthetic demo case body text.",
    )


def test_new_case_id_and_trace_id_are_unique_and_prefixed() -> None:
    ids = {new_case_id() for _ in range(20)}
    trace_ids = {new_trace_id() for _ in range(20)}
    assert len(ids) == 20
    assert len(trace_ids) == 20
    assert all(i.startswith("case_") for i in ids)
    assert all(t.startswith("trace_") for t in trace_ids)


def test_create_initial_state_defaults() -> None:
    state = create_initial_state(email=_demo_email())

    assert state["goal"] is None
    assert state["goal_status"] == GoalStatus.CREATED
    assert state["success_criteria"] == []
    assert state["execution_plan"] is None
    assert state["plan_history"] == []
    assert state["case_id"].startswith("case_")
    assert state["trace_id"].startswith("trace_")
    assert state["email"] is not None
    assert state["email"].subject.startswith("Possible adverse event report")
    assert state["attachments"] == []
    assert state["agent_events"] == []
    assert state["tool_events"] == []
    assert state["memory_events"] == []
    assert state["retrieval_events"] == []
    assert state["errors"] == []
    assert state["retry_counts"] == {}
    assert state["total_cost"] == 0.0
    assert state["model_version"] == "phase1-scaffold"


def test_create_initial_state_accepts_explicit_ids() -> None:
    state = create_initial_state(
        email=_demo_email(), case_id="case_fixed_001", trace_id="trace_fixed_001"
    )
    assert state["case_id"] == "case_fixed_001"
    assert state["trace_id"] == "trace_fixed_001"


def test_agent_events_reducer_merges_across_partial_updates() -> None:
    """Simulate what LangGraph does when two node updates both touch
    agent_events: with an operator.add reducer, results are concatenated,
    never overwritten — required so no agent's audit trail is lost.
    """
    state = create_initial_state(email=_demo_email())

    update_a = [
        AgentEvent(
            case_id=state["case_id"],
            trace_id=state["trace_id"],
            agent=AgentName.GOAL_AGENT,
            node="goal_manager",
            decision_summary="Goal created.",
        )
    ]
    update_b = [
        AgentEvent(
            case_id=state["case_id"],
            trace_id=state["trace_id"],
            agent=AgentName.PLANNER,
            node="planner",
            decision_summary="Plan created with 12 steps.",
        )
    ]

    merged = operator.add(state["agent_events"], update_a)
    merged = operator.add(merged, update_b)

    assert len(merged) == 2
    assert merged[0].agent == AgentName.GOAL_AGENT
    assert merged[1].agent == AgentName.PLANNER
