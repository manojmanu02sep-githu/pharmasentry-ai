"""Workflow tools: case-state access, human-review pause, routing, and audit.

`read_case_state`/`write_case_checkpoint` enforce case isolation: a call
may only touch the case named in its own `ToolContext.case_id`. Any
mismatch — an agent for case A asking to read/write case B's state — is a
cross-case access attempt and is rejected via `ToolAuthorizationError`
(no retry), not a soft failure. Storage is behind a small `CaseStateStore`
protocol; the in-memory default here is swapped for the real (episodic
memory / LangGraph checkpointer) backend in Phase 4/8 without changing
this tool's interface.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from src.models.enums import (
    AgentName,
    FileValidationStatus,
    OCRQuality,
    RouteReason,
    ToolCallStatus,
    TriagePriority,
)
from src.models.review import ReviewTask
from src.tools.base import BaseTool, ToolAuthorizationError, ToolContext


class CaseStateStore(Protocol):
    def get(self, case_id: str) -> dict[str, Any] | None: ...
    def put(self, case_id: str, data: dict[str, Any]) -> None: ...


class InMemoryCaseStateStore:
    """Process-local store. Fine for tests and the local demo; Phase 4
    adds a SQLite-backed episodic-memory store behind the same protocol."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def get(self, case_id: str) -> dict[str, Any] | None:
        return self._data.get(case_id)

    def put(self, case_id: str, data: dict[str, Any]) -> None:
        self._data[case_id] = data


def _enforce_case_isolation(target_case_id: str, ctx: ToolContext) -> None:
    if target_case_id != ctx.case_id:
        raise ToolAuthorizationError(
            f"cross-case access blocked: agent for case {ctx.case_id!r} "
            f"requested case {target_case_id!r}"
        )


# --- read_case_state --------------------------------------------------------


class ReadCaseStateInput(BaseModel):
    case_id: str


class ReadCaseStateOutput(BaseModel):
    case_id: str
    found: bool
    data: dict[str, Any] = Field(default_factory=dict)


class ReadCaseStateTool(BaseTool[ReadCaseStateInput, ReadCaseStateOutput]):
    name = "read_case_state"
    purpose = "Read the checkpointed state for the CALLER'S OWN case only."
    timeout_seconds = 2.0
    max_retries = 2

    def __init__(self, store: CaseStateStore | None = None):
        self.store = store or InMemoryCaseStateStore()

    def _execute(
        self, tool_input: ReadCaseStateInput, ctx: ToolContext
    ) -> ReadCaseStateOutput:
        _enforce_case_isolation(tool_input.case_id, ctx)
        data = self.store.get(tool_input.case_id)
        return ReadCaseStateOutput(
            case_id=tool_input.case_id, found=data is not None, data=data or {}
        )


# --- write_case_checkpoint --------------------------------------------------------


class WriteCaseCheckpointInput(BaseModel):
    case_id: str
    checkpoint_data: dict[str, Any]


class WriteCaseCheckpointOutput(BaseModel):
    case_id: str
    written: bool


class WriteCaseCheckpointTool(BaseTool[WriteCaseCheckpointInput, WriteCaseCheckpointOutput]):
    name = "write_case_checkpoint"
    purpose = "Persist a checkpoint for the CALLER'S OWN case only."
    timeout_seconds = 2.0
    max_retries = 0  # a write must never silently run twice

    def __init__(self, store: CaseStateStore | None = None):
        self.store = store or InMemoryCaseStateStore()

    def _execute(
        self, tool_input: WriteCaseCheckpointInput, ctx: ToolContext
    ) -> WriteCaseCheckpointOutput:
        _enforce_case_isolation(tool_input.case_id, ctx)
        self.store.put(tool_input.case_id, tool_input.checkpoint_data)
        return WriteCaseCheckpointOutput(case_id=tool_input.case_id, written=True)


# --- pause_for_human_review --------------------------------------------------------


class PauseForHumanReviewInput(BaseModel):
    case_id: str
    reason: str
    evidence_summary: str = Field(max_length=2000)


class PauseForHumanReviewOutput(BaseModel):
    case_id: str
    status: str = "paused"
    review_task_id: str


class PauseForHumanReviewTool(BaseTool[PauseForHumanReviewInput, PauseForHumanReviewOutput]):
    name = "pause_for_human_review"
    purpose = (
        "Record that graph execution has paused for this case and is "
        "awaiting an explicit human decision. Resume happens only via the "
        "Human Review Controller, never automatically."
    )
    timeout_seconds = 2.0
    max_retries = 0  # side-effecting: must not create two pause records for one call

    def _execute(
        self, tool_input: PauseForHumanReviewInput, ctx: ToolContext
    ) -> PauseForHumanReviewOutput:
        _enforce_case_isolation(tool_input.case_id, ctx)
        return PauseForHumanReviewOutput(
            case_id=tool_input.case_id,
            review_task_id=f"task_{uuid.uuid4().hex[:12]}",
        )


# --- route_case --------------------------------------------------------


class RouteCaseInput(BaseModel):
    file_validation_status: FileValidationStatus = FileValidationStatus.VALID
    ocr_quality: OCRQuality = OCRQuality.NOT_APPLICABLE
    is_safety_content: bool = True
    llm_available: bool = True
    minimum_criteria_met: bool = True
    has_conflicting_fields: bool = False
    has_duplicate_candidate: bool = False
    has_triage_indicator: bool = False  # informational only; does not change the route


class RouteCaseOutput(BaseModel):
    route: RouteReason
    reason: str


class RouteCaseTool(BaseTool[RouteCaseInput, RouteCaseOutput]):
    name = "route_case"
    purpose = (
        "Deterministic conditional routing per CLAUDE.md's Conditional "
        "routes list. A pure decision table over already-computed signals "
        "— it never itself infers seriousness/duplication/criteria."
    )
    timeout_seconds = 1.0
    max_retries = 2

    def _execute(self, tool_input: RouteCaseInput, ctx: ToolContext) -> RouteCaseOutput:
        i = tool_input
        if i.file_validation_status != FileValidationStatus.VALID:
            return RouteCaseOutput(
                route=RouteReason.UNSUPPORTED_FILE,
                reason=f"file_validation_status={i.file_validation_status.value}",
            )
        if i.ocr_quality in (OCRQuality.DEGRADED, OCRQuality.UNREADABLE):
            return RouteCaseOutput(
                route=RouteReason.LOW_OCR_QUALITY, reason=f"ocr_quality={i.ocr_quality.value}"
            )
        if not i.is_safety_content:
            return RouteCaseOutput(
                route=RouteReason.NON_SAFETY_CONTENT, reason="non-safety content"
            )
        if not i.llm_available:
            return RouteCaseOutput(
                route=RouteReason.LLM_UNAVAILABLE, reason="LLM provider unavailable"
            )
        if not i.minimum_criteria_met:
            return RouteCaseOutput(
                route=RouteReason.MISSING_MINIMUM_CRITERIA, reason="minimum criteria not met"
            )
        if i.has_conflicting_fields:
            return RouteCaseOutput(
                route=RouteReason.VALIDATION_FAILURE, reason="conflicting field values"
            )
        if i.has_duplicate_candidate:
            return RouteCaseOutput(
                route=RouteReason.POTENTIAL_DUPLICATE, reason="potential duplicate candidate found"
            )
        return RouteCaseOutput(route=RouteReason.NORMAL, reason="no routing exception triggered")


# --- record_audit_event --------------------------------------------------------


class RecordAuditEventInput(BaseModel):
    case_id: str
    trace_id: str
    agent: AgentName
    action_name: str
    status: ToolCallStatus = ToolCallStatus.SUCCESS
    summary: str = Field(max_length=500)


class RecordAuditEventOutput(BaseModel):
    case_id: str
    trace_id: str
    action_name: str
    status: ToolCallStatus
    summary: str
    recorded_at: datetime


class RecordAuditEventTool(BaseTool[RecordAuditEventInput, RecordAuditEventOutput]):
    name = "record_audit_event"
    purpose = (
        "Construct a structured, concise audit entry for an agent action "
        "that isn't itself another tool call (e.g. a decision point). "
        "Never records chain-of-thought — only the caller-provided summary."
    )
    timeout_seconds = 1.0
    max_retries = 2  # constructing a record is side-effect-free (caller appends it)

    def _execute(
        self, tool_input: RecordAuditEventInput, ctx: ToolContext
    ) -> RecordAuditEventOutput:
        return RecordAuditEventOutput(
            case_id=tool_input.case_id,
            trace_id=tool_input.trace_id,
            action_name=tool_input.action_name,
            status=tool_input.status,
            summary=tool_input.summary,
            recorded_at=datetime.now(UTC),
        )


# --- create_review_task --------------------------------------------------------


class CreateReviewTaskInput(BaseModel):
    case_id: str
    reason: str
    evidence_summary: str = Field(max_length=2000)
    ai_suggestion: str | None = None
    ai_suggested_priority: TriagePriority | None = None


class CreateReviewTaskTool(BaseTool[CreateReviewTaskInput, ReviewTask]):
    name = "create_review_task"
    purpose = "Create a queued Human Review Queue entry for a case."
    timeout_seconds = 1.0
    max_retries = 0  # side-effecting: must not create duplicate tasks

    def _execute(self, tool_input: CreateReviewTaskInput, ctx: ToolContext) -> ReviewTask:
        _enforce_case_isolation(tool_input.case_id, ctx)
        return ReviewTask(
            task_id=f"review_{uuid.uuid4().hex[:12]}",
            case_id=tool_input.case_id,
            reason=tool_input.reason,
            evidence_summary=tool_input.evidence_summary,
            ai_suggestion=tool_input.ai_suggestion,
            ai_suggested_priority=tool_input.ai_suggested_priority,
        )
