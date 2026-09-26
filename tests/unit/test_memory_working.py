"""Unit tests for src/memory/working.py."""

from __future__ import annotations

import pytest

from src.graph.state import create_initial_state
from src.memory.working import read_working_memory, write_working_memory
from src.models import EmailMessage
from src.models.enums import AgentName, MemoryOperation, MemoryTier
from src.tools.base import ToolAuthorizationError, ToolContext


def _email() -> EmailMessage:
    return EmailMessage(
        message_id="msg_001",
        sender="r.tanaka@fictionalclinic-demo.example",
        subject="Synthetic demo case",
        body="Synthetic demo case body text.",
    )


def _ctx(case_id: str) -> ToolContext:
    return ToolContext(
        case_id=case_id,
        trace_id="trace_001",
        agent=AgentName.SUPERVISOR,
        authorized_tools=frozenset(),
    )


def test_write_then_read_round_trip() -> None:
    state = create_initial_state(email=_email(), case_id="case_a")
    ctx = _ctx("case_a")

    write_event = write_working_memory(state, "current_step", "intake", ctx)
    value, read_event = read_working_memory(state, "current_step", ctx)

    assert value == "intake"
    assert state["current_step"] == "intake"
    assert write_event.tier == MemoryTier.WORKING
    assert write_event.operation == MemoryOperation.WRITE
    assert read_event.operation == MemoryOperation.READ


def test_read_missing_key_returns_none() -> None:
    state = create_initial_state(email=_email(), case_id="case_a")
    value, _ = read_working_memory(state, "no_such_field", _ctx("case_a"))
    assert value is None


def test_cross_case_read_is_blocked() -> None:
    state = create_initial_state(email=_email(), case_id="case_a")
    with pytest.raises(ToolAuthorizationError):
        read_working_memory(state, "current_step", _ctx("case_b"))


def test_cross_case_write_is_blocked() -> None:
    state = create_initial_state(email=_email(), case_id="case_a")
    with pytest.raises(ToolAuthorizationError):
        write_working_memory(state, "current_step", "tampered", _ctx("case_b"))
    assert state["current_step"] is None  # the blocked write never landed
