"""Unit tests for src/tools/workflow.py — especially case-isolation enforcement."""

from __future__ import annotations

import pytest

from src.models.enums import AgentName, FileValidationStatus, OCRQuality, RouteReason
from src.tools.base import ToolAuthorizationError, ToolContext
from src.tools.workflow import (
    CreateReviewTaskInput,
    CreateReviewTaskTool,
    InMemoryCaseStateStore,
    PauseForHumanReviewInput,
    PauseForHumanReviewTool,
    ReadCaseStateInput,
    ReadCaseStateTool,
    RecordAuditEventInput,
    RecordAuditEventTool,
    RouteCaseInput,
    RouteCaseTool,
    WriteCaseCheckpointInput,
    WriteCaseCheckpointTool,
)

CTX_CASE_A = ToolContext(
    case_id="case_a",
    trace_id="trace_001",
    agent=AgentName.SUPERVISOR,
    authorized_tools=frozenset(
        {
            "read_case_state", "write_case_checkpoint", "pause_for_human_review",
            "route_case", "record_audit_event", "create_review_task",
        }
    ),
)
CTX_CASE_B = CTX_CASE_A.model_copy(update={"case_id": "case_b"})


def test_write_then_read_case_state_round_trip() -> None:
    store = InMemoryCaseStateStore()
    write_tool = WriteCaseCheckpointTool(store=store)
    read_tool = ReadCaseStateTool(store=store)

    write_out, _ = write_tool.run(
        WriteCaseCheckpointInput(case_id="case_a", checkpoint_data={"step": "intake"}),
        CTX_CASE_A,
    )
    assert write_out.written is True

    read_out, _ = read_tool.run(ReadCaseStateInput(case_id="case_a"), CTX_CASE_A)
    assert read_out.found is True
    assert read_out.data == {"step": "intake"}


def test_read_case_state_missing_case_returns_not_found() -> None:
    read_tool = ReadCaseStateTool(store=InMemoryCaseStateStore())
    output, _ = read_tool.run(ReadCaseStateInput(case_id="case_a"), CTX_CASE_A)
    assert output.found is False
    assert output.data == {}


def test_cross_case_read_is_blocked() -> None:
    store = InMemoryCaseStateStore()
    store.put("case_a", {"secret": "case A data"})
    read_tool = ReadCaseStateTool(store=store)

    with pytest.raises(ToolAuthorizationError):
        # CTX_CASE_B's agent is processing case_b but asks to read case_a
        read_tool.run(ReadCaseStateInput(case_id="case_a"), CTX_CASE_B)


def test_cross_case_write_is_blocked() -> None:
    store = InMemoryCaseStateStore()
    write_tool = WriteCaseCheckpointTool(store=store)

    with pytest.raises(ToolAuthorizationError):
        write_tool.run(
            WriteCaseCheckpointInput(case_id="case_a", checkpoint_data={"tampered": True}),
            CTX_CASE_B,
        )
    assert store.get("case_a") is None  # the blocked write never landed


def test_cross_case_pause_and_review_task_are_blocked() -> None:
    with pytest.raises(ToolAuthorizationError):
        PauseForHumanReviewTool().run(
            PauseForHumanReviewInput(case_id="case_a", reason="x", evidence_summary="y"),
            CTX_CASE_B,
        )
    with pytest.raises(ToolAuthorizationError):
        CreateReviewTaskTool().run(
            CreateReviewTaskInput(case_id="case_a", reason="x", evidence_summary="y"),
            CTX_CASE_B,
        )


def test_pause_for_human_review_returns_a_task_id() -> None:
    output, event = PauseForHumanReviewTool().run(
        PauseForHumanReviewInput(
            case_id="case_a", reason="seriousness confirmation needed",
            evidence_summary="hospitalization mentioned",
        ),
        CTX_CASE_A,
    )
    assert output.status == "paused"
    assert output.review_task_id.startswith("task_")
    assert event.case_id == "case_a"


def test_create_review_task_shape() -> None:
    task, _ = CreateReviewTaskTool().run(
        CreateReviewTaskInput(
            case_id="case_a", reason="duplicate review", evidence_summary="two similar cases",
            ai_suggestion="possible duplicate",
        ),
        CTX_CASE_A,
    )
    assert task.case_id == "case_a"
    assert task.status == "pending"
    assert task.ai_suggestion == "possible duplicate"


def test_record_audit_event_shape() -> None:
    output, _ = RecordAuditEventTool().run(
        RecordAuditEventInput(
            case_id="case_a", trace_id="trace_001", agent=AgentName.SUPERVISOR,
            action_name="delegate_to_intake", summary="Delegated to Intake Agent.",
        ),
        CTX_CASE_A,
    )
    assert output.action_name == "delegate_to_intake"
    assert output.case_id == "case_a"


def test_route_case_unsupported_file_takes_priority() -> None:
    output, _ = RouteCaseTool().run(
        RouteCaseInput(
            file_validation_status=FileValidationStatus.UNSUPPORTED_TYPE,
            ocr_quality=OCRQuality.UNREADABLE,  # would also trigger, but file check wins
        ),
        CTX_CASE_A,
    )
    assert output.route == RouteReason.UNSUPPORTED_FILE


def test_route_case_low_ocr_quality() -> None:
    output, _ = RouteCaseTool().run(
        RouteCaseInput(ocr_quality=OCRQuality.DEGRADED), CTX_CASE_A
    )
    assert output.route == RouteReason.LOW_OCR_QUALITY


def test_route_case_non_safety_content() -> None:
    output, _ = RouteCaseTool().run(RouteCaseInput(is_safety_content=False), CTX_CASE_A)
    assert output.route == RouteReason.NON_SAFETY_CONTENT


def test_route_case_llm_unavailable() -> None:
    output, _ = RouteCaseTool().run(RouteCaseInput(llm_available=False), CTX_CASE_A)
    assert output.route == RouteReason.LLM_UNAVAILABLE


def test_route_case_missing_minimum_criteria() -> None:
    output, _ = RouteCaseTool().run(
        RouteCaseInput(minimum_criteria_met=False), CTX_CASE_A
    )
    assert output.route == RouteReason.MISSING_MINIMUM_CRITERIA


def test_route_case_conflicting_fields() -> None:
    output, _ = RouteCaseTool().run(
        RouteCaseInput(has_conflicting_fields=True), CTX_CASE_A
    )
    assert output.route == RouteReason.VALIDATION_FAILURE


def test_route_case_potential_duplicate() -> None:
    output, _ = RouteCaseTool().run(
        RouteCaseInput(has_duplicate_candidate=True), CTX_CASE_A
    )
    assert output.route == RouteReason.POTENTIAL_DUPLICATE


def test_route_case_normal_when_nothing_flagged() -> None:
    output, _ = RouteCaseTool().run(RouteCaseInput(), CTX_CASE_A)
    assert output.route == RouteReason.NORMAL


def test_route_case_seriousness_indicator_alone_does_not_change_route() -> None:
    output, _ = RouteCaseTool().run(
        RouteCaseInput(has_seriousness_indicator=True), CTX_CASE_A
    )
    assert output.route == RouteReason.NORMAL
