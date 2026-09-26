"""Working memory: logged, case-isolated access to the in-flight CaseState.

Working memory IS the LangGraph CaseState (src/graph/state.py) — this module
does not introduce a second, parallel store. It wraps state reads/writes so
every access during a run enforces case isolation and produces a
MemoryEvent, per CLAUDE.md's Memory section ("Enforce case isolation,
authorized access, retention controls, and logged reads/writes").
"""

from __future__ import annotations

from typing import Any, cast

from src.graph.state import CaseState
from src.models.audit import MemoryEvent
from src.models.enums import MemoryOperation, MemoryTier
from src.tools.base import ToolAuthorizationError, ToolContext


def _enforce_case_isolation(state: CaseState, ctx: ToolContext) -> None:
    if state["case_id"] != ctx.case_id:
        raise ToolAuthorizationError(
            f"cross-case working-memory access blocked: state belongs to "
            f"{state['case_id']!r}, agent for case {ctx.case_id!r} requested it"
        )


def read_working_memory(state: CaseState, key: str, ctx: ToolContext) -> tuple[Any, MemoryEvent]:
    """Read one field from the caller's own in-flight CaseState."""
    _enforce_case_isolation(state, ctx)
    value = cast("dict[str, Any]", state).get(key)
    event = MemoryEvent(
        case_id=ctx.case_id,
        trace_id=ctx.trace_id,
        tier=MemoryTier.WORKING,
        operation=MemoryOperation.READ,
        key=key,
        agent=ctx.agent,
    )
    return value, event


def write_working_memory(
    state: CaseState, key: str, value: Any, ctx: ToolContext
) -> MemoryEvent:
    """Write one field into the caller's own in-flight CaseState."""
    _enforce_case_isolation(state, ctx)
    cast("dict[str, Any]", state)[key] = value
    return MemoryEvent(
        case_id=ctx.case_id,
        trace_id=ctx.trace_id,
        tier=MemoryTier.WORKING,
        operation=MemoryOperation.WRITE,
        key=key,
        agent=ctx.agent,
    )
