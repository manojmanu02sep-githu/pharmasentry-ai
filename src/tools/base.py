"""Shared tool execution harness.

Every tool under src/tools/ is a `BaseTool` subclass. CLAUDE.md requires
every tool to have typed input/output, an authorization policy, input
validation, a timeout, retry configuration where safe, a result-size
limit, a structured audit event, and safe error handling. Implementing
each of those seven times per tool (35 tools) would be exactly the kind
of duplication the project rules warn against, so this module implements
them ONCE; individual tool classes only implement `_execute` and declare
a few class attributes.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import UTC, datetime
from typing import Generic, TypeVar

from pydantic import BaseModel

from src.models.audit import ToolEvent
from src.models.enums import AgentName, ToolCallStatus

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)
T = TypeVar("T")

_MAX_SUMMARY_CHARS = 200


class ToolAuthorizationError(RuntimeError):
    """Raised when an agent calls a tool that is not on its allowlist, or
    attempts to access another case's data (cross-case access)."""


class ToolExecutionError(RuntimeError):
    """Raised when a tool fails after exhausting its retries.

    The message is a short, sanitized description — no raw exception
    internals and no case content — safe to surface in logs or the UI.
    The triggering ToolEvent (status=ERROR/TIMEOUT) is attached as
    `.tool_event` so callers can still append it to the audit trail.
    """

    def __init__(self, message: str, tool_event: ToolEvent):
        super().__init__(message)
        self.tool_event = tool_event


class ToolContext(BaseModel):
    """Everything a tool call needs to know about who is calling it.

    `authorized_tools` is normally derived from config/config.yaml's
    per-agent tool_allowlist (see src/tools/registry.py) — kept as a plain
    frozenset here so tests can construct a ToolContext without reading
    the on-disk config.
    """

    case_id: str
    trace_id: str
    agent: AgentName
    authorized_tools: frozenset[str]


def _summarize(model: BaseModel | None, max_len: int = _MAX_SUMMARY_CHARS) -> str:
    """A short, safe-to-log summary of a Pydantic model.

    Never logs complete case text by default (Observability requirement) —
    truncates aggressively rather than dumping full field values.
    """
    if model is None:
        return ""
    text = model.model_dump_json()
    return text if len(text) <= max_len else text[:max_len] + "...(truncated)"


def limit_results(items: list[T], limit: int) -> tuple[list[T], bool]:
    """Cap a list-shaped tool output at `limit` items.

    Returns (possibly-truncated list, whether truncation occurred) so the
    caller's output model can carry an honest `truncated` flag instead of
    silently dropping results.
    """
    if len(items) <= limit:
        return items, False
    return items[:limit], True


_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="pharmasentry-tool")


class BaseTool(ABC, Generic[InputT, OutputT]):
    """Base class every deterministic tool extends.

    Subclasses set `name`, `purpose`, `timeout_seconds`, and `max_retries`
    as class attributes and implement `_execute`. Everything else —
    authorization, timeout enforcement, bounded retry, audit event
    construction, and safe error wrapping — is handled by `run()`.
    """

    name: str
    purpose: str
    timeout_seconds: float = 5.0
    # Only safe (read-only / idempotent) tools should retry. Side-effecting
    # tools (audit logging, checkpoint writes, review-task creation) must
    # keep this at 0 so a transient failure never runs twice.
    max_retries: int = 0

    @abstractmethod
    def _execute(self, tool_input: InputT, ctx: ToolContext) -> OutputT:
        """The tool's actual logic. Raise on genuine failure; return a
        normal (possibly negative/failed-validation) output for expected
        business outcomes — e.g. an unsupported file is a valid
        FileValidationOutput(status=UNSUPPORTED_TYPE), not an exception."""

    def _authorize(self, ctx: ToolContext) -> None:
        if self.name not in ctx.authorized_tools:
            raise ToolAuthorizationError(
                f"agent {ctx.agent.value!r} is not authorized to call tool {self.name!r}"
            )

    def run(self, tool_input: InputT, ctx: ToolContext) -> tuple[OutputT, ToolEvent]:
        self._authorize(ctx)  # pydantic already validated tool_input's shape at construction

        attempt = 0
        last_error: BaseException | None = None
        status = ToolCallStatus.ERROR
        start = time.monotonic()

        while attempt <= self.max_retries:
            try:
                future = _executor.submit(self._execute, tool_input, ctx)
                output = future.result(timeout=self.timeout_seconds)
            except FutureTimeoutError as exc:
                last_error = exc
                status = ToolCallStatus.TIMEOUT
                attempt += 1
                continue
            except ToolAuthorizationError:
                raise
            except Exception as exc:  # noqa: BLE001 - converted to a safe error below
                last_error = exc
                status = ToolCallStatus.ERROR
                attempt += 1
                continue

            latency_ms = (time.monotonic() - start) * 1000
            event = ToolEvent(
                case_id=ctx.case_id,
                trace_id=ctx.trace_id,
                tool_name=self.name,
                agent=ctx.agent,
                status=ToolCallStatus.RETRIED if attempt > 0 else ToolCallStatus.SUCCESS,
                input_summary=_summarize(tool_input),
                output_summary=_summarize(output),
                latency_ms=latency_ms,
                authorized=True,
                timestamp=datetime.now(UTC),
            )
            return output, event

        latency_ms = (time.monotonic() - start) * 1000
        event = ToolEvent(
            case_id=ctx.case_id,
            trace_id=ctx.trace_id,
            tool_name=self.name,
            agent=ctx.agent,
            status=status,
            input_summary=_summarize(tool_input),
            output_summary=f"failed after {attempt} attempt(s): {type(last_error).__name__}",
            latency_ms=latency_ms,
            authorized=True,
            timestamp=datetime.now(UTC),
        )
        raise ToolExecutionError(
            f"{self.name} failed after {attempt} attempt(s)", tool_event=event
        ) from last_error


def run_tool(
    tool: BaseTool[InputT, OutputT],
    tool_input: InputT,
    ctx: ToolContext,
    audit_sink: Callable[[ToolEvent], None] | None = None,
) -> OutputT:
    """Convenience wrapper: run a tool, optionally push its ToolEvent to an
    audit sink (e.g. state["tool_events"].append), and return just the
    output. Used by agents that don't need the ToolEvent directly."""
    output, event = tool.run(tool_input, ctx)
    if audit_sink is not None:
        audit_sink(event)
    return output
