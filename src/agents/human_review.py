"""Human Review Controller: pauses the case and creates a review task.
This is the mandatory gate -- CLAUDE.md's Absolute Boundaries require
every consequential output to have human approval, so this node never
auto-approves, auto-rejects, or advances goal_status to COMPLETED itself;
it only records that the case is now AWAITING_HUMAN and exposes a
ReviewTask for a reviewer to act on via the Case Workspace / Human Review
Queue UI, which resumes the checkpointed graph after a real decision."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from src.agents._common import call_tool, make_agent_event, make_tool_context
from src.models.enums import AgentName, DecisionOutcome, GoalStatus, ReviewDecision, ToolCallStatus
from src.models.goal import SuccessCriterion
from src.models.reasoning import AgentDecision
from src.models.review import ReviewerChange, ReviewStatus
from src.tools.workflow import (
    CreateReviewTaskInput,
    PauseForHumanReviewInput,
    RecordAuditEventInput,
)

NODE = AgentName.HUMAN_REVIEW


def _build_evidence_summary(state: dict) -> str:
    triage = state.get("triage_result")
    duplicates = state.get("duplicate_candidates") or []
    missing = state.get("missing_information") or []
    parts = [
        f"{len(triage.findings) if triage else 0} seriousness indicator(s)",
        f"{len(duplicates)} duplicate candidate(s)",
        f"{len(missing)} missing field(s)",
    ]
    return "; ".join(parts)[:2000]


def run(state: dict) -> dict:
    started = time.monotonic()
    case_id = state["case_id"]
    trace_id = state["trace_id"]
    ctx = make_tool_context(case_id, trace_id, NODE)
    tool_events: list = []

    evidence_summary = _build_evidence_summary(state)
    triage = state.get("triage_result")
    reason = (
        "Seriousness indicator(s) detected; requires confirmation."
        if triage and triage.any_indicator_present
        else "Preliminary case package ready for authorized human review."
    )

    pause_output = call_tool(
        "pause_for_human_review",
        PauseForHumanReviewInput(case_id=case_id, reason=reason, evidence_summary=evidence_summary),
        ctx,
        tool_events,
    )
    review_task = call_tool(
        "create_review_task",
        CreateReviewTaskInput(
            case_id=case_id,
            reason=reason,
            evidence_summary=evidence_summary,
            ai_suggestion=None,
            ai_suggested_priority=triage.suggested_priority if triage else None,
        ),
        ctx,
        tool_events,
    )

    updated_criteria = [
        SuccessCriterion(
            criterion_id=c.criterion_id,
            description=c.description,
            met=True if c.criterion_id == "human_review_reached" else c.met,
            evidence_ref=c.evidence_ref,
        )
        for c in state.get("success_criteria") or []
    ]
    review_status = ReviewStatus(case_id=case_id, decision=ReviewDecision.PENDING)

    call_tool(
        "record_audit_event",
        RecordAuditEventInput(
            case_id=case_id,
            trace_id=trace_id,
            agent=NODE,
            action_name="paused_for_review",
            status=ToolCallStatus.SUCCESS,
            summary=f"Paused; review_task_id={review_task.task_id}.",
        ),
        ctx,
        tool_events,
    )

    decision = AgentDecision(
        agent=NODE,
        case_id=case_id,
        decision="paused",
        evidence=[evidence_summary],
        confidence=1.0,
        decision_summary=f"Case paused for human review (task {review_task.task_id}); "
        f"status={pause_output.status}. No automated approval, merge, or regulatory action taken.",
        next_action=DecisionOutcome.STOP,
        requires_human_review=True,
    )
    latency_ms = (time.monotonic() - started) * 1000
    agent_event = make_agent_event(
        decision,
        trace_id,
        node="human_review",
        model_version=state["model_version"],
        latency_ms=latency_ms,
    )

    return {
        "review_status": review_status,
        "success_criteria": updated_criteria,
        "goal_status": GoalStatus.AWAITING_HUMAN,
        "current_step": "human_review",
        "agent_events": [agent_event],
        "tool_events": tool_events,
    }


def apply_review_decision(
    state: dict,
    decision: ReviewDecision,
    reviewer_id: str,
    notes: str | None = None,
    field_edits: list[ReviewerChange] | None = None,
) -> dict:
    """Record an authorized reviewer's decision. This is the ONLY place a
    case moves out of AWAITING_HUMAN -- called directly by the UI/API
    layer, never by the graph itself, so a human approval can never be
    bypassed by automated re-invocation of the pipeline."""
    if state.get("goal_status") != GoalStatus.AWAITING_HUMAN:
        raise ValueError("Case is not awaiting human review; cannot apply a review decision.")

    review_status = ReviewStatus(
        case_id=state["case_id"],
        decision=decision,
        reviewer_id=reviewer_id,
        notes=notes,
        decided_at=datetime.now(UTC),
    )
    goal_status = {
        ReviewDecision.APPROVED: GoalStatus.COMPLETED,
        ReviewDecision.REJECTED: GoalStatus.REJECTED,
        ReviewDecision.CHANGES_REQUESTED: GoalStatus.AWAITING_HUMAN,
        ReviewDecision.PENDING: GoalStatus.AWAITING_HUMAN,
    }[decision]

    return {
        "review_status": review_status,
        "reviewer_changes": field_edits or [],
        "goal_status": goal_status,
    }
