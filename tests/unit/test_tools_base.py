"""Unit tests for the shared tool execution harness (src/tools/base.py)."""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from src.models.enums import AgentName, ToolCallStatus
from src.tools.base import (
    BaseTool,
    ToolAuthorizationError,
    ToolContext,
    ToolExecutionError,
    limit_results,
)


class _Input(BaseModel):
    value: int


class _Output(BaseModel):
    doubled: int


class _DoubleTool(BaseTool[_Input, _Output]):
    name = "double_tool"
    purpose = "test tool: doubles a number"
    timeout_seconds = 1.0
    max_retries = 0

    def _execute(self, tool_input: _Input, ctx: ToolContext) -> _Output:
        return _Output(doubled=tool_input.value * 2)


class _AlwaysFailsTool(BaseTool[_Input, _Output]):
    name = "always_fails_tool"
    purpose = "test tool: always raises"
    timeout_seconds = 1.0
    max_retries = 2

    def __init__(self) -> None:
        self.calls = 0

    def _execute(self, tool_input: _Input, ctx: ToolContext) -> _Output:
        self.calls += 1
        raise ValueError("boom")


class _SlowTool(BaseTool[_Input, _Output]):
    name = "slow_tool"
    purpose = "test tool: sleeps past its own timeout"
    timeout_seconds = 0.05
    max_retries = 0

    def _execute(self, tool_input: _Input, ctx: ToolContext) -> _Output:
        time.sleep(0.5)
        return _Output(doubled=tool_input.value * 2)


_DEFAULT_AUTHORIZED = frozenset({"double_tool", "always_fails_tool", "slow_tool"})


def _ctx(authorized: frozenset[str] = _DEFAULT_AUTHORIZED) -> ToolContext:
    return ToolContext(
        case_id="case_001",
        trace_id="trace_001",
        agent=AgentName.SUBJECT_READER,
        authorized_tools=authorized,
    )


def test_successful_run_returns_output_and_success_event() -> None:
    tool = _DoubleTool()
    output, event = tool.run(_Input(value=21), _ctx())
    assert output.doubled == 42
    assert event.status == ToolCallStatus.SUCCESS
    assert event.tool_name == "double_tool"
    assert event.case_id == "case_001"
    assert event.authorized is True


def test_unauthorized_tool_raises_immediately_without_retry() -> None:
    tool = _AlwaysFailsTool()
    with pytest.raises(ToolAuthorizationError):
        tool.run(_Input(value=1), _ctx(authorized=frozenset()))
    assert tool.calls == 0  # never even attempted


def test_failing_tool_retries_up_to_max_retries_then_raises() -> None:
    tool = _AlwaysFailsTool()
    with pytest.raises(ToolExecutionError) as exc_info:
        tool.run(_Input(value=1), _ctx())
    assert tool.calls == 3  # 1 initial + 2 retries
    assert exc_info.value.tool_event.status == ToolCallStatus.ERROR
    # the sanitized error message never leaks the raw ValueError text
    assert "boom" not in str(exc_info.value)


def test_timeout_raises_tool_execution_error_with_timeout_event() -> None:
    tool = _SlowTool()
    with pytest.raises(ToolExecutionError) as exc_info:
        tool.run(_Input(value=1), _ctx())
    assert exc_info.value.tool_event.status == ToolCallStatus.TIMEOUT


def test_input_summary_is_truncated_and_safe() -> None:
    tool = _DoubleTool()
    _, event = tool.run(_Input(value=7), _ctx())
    assert len(event.input_summary) <= 220  # small model, well under the cap
    assert "value" in event.input_summary


def test_limit_results_truncates_and_flags() -> None:
    items, truncated = limit_results([1, 2, 3, 4, 5], 3)
    assert items == [1, 2, 3]
    assert truncated is True

    items, truncated = limit_results([1, 2], 3)
    assert items == [1, 2]
    assert truncated is False
