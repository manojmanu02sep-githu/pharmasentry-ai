"""Typed LangGraph shared state (working memory) for one case.

Every field listed under "Shared State" in CLAUDE.md is represented here.
List-valued fields that multiple nodes append to (events, history, errors)
use ``operator.add`` reducers so LangGraph merges partial node updates
instead of overwriting them.
"""

from __future__ import annotations

import operator
import uuid
from datetime import UTC, datetime
from typing import Annotated, TypedDict

from src.models import (
    AgentEvent,
    Attachment,
    CaseGoal,
    EmailMessage,
    ErrorRecord,
    EvaluationResult,
    ExecutionPlan,
    ExtractedFields,
    FollowUpDraft,
    GoalStatus,
    MemoryEvent,
    MinimumCriteriaResult,
    NarrativeDraft,
    PlanRevision,
    RetrievalEvent,
    ReviewerChange,
    ReviewStatus,
    SeriousnessTriageResult,
    SourcePassage,
    SuccessCriterion,
    ToolEvent,
)
from src.models.duplicates import DuplicateCandidate
from src.models.missing_info import MissingInformationItem


class CaseState(TypedDict):
    # Goal
    goal: CaseGoal | None
    success_criteria: list[SuccessCriterion]
    goal_status: GoalStatus

    # Planning
    execution_plan: ExecutionPlan | None
    plan_history: Annotated[list[PlanRevision], operator.add]
    current_step: str | None

    # Identity / trace
    case_id: str
    trace_id: str

    # Intake evidence
    email: EmailMessage | None
    attachments: list[Attachment]
    source_passages: list[SourcePassage]

    # Extraction and downstream analysis
    extracted_fields: ExtractedFields | None
    minimum_criteria: MinimumCriteriaResult | None
    seriousness_triage: SeriousnessTriageResult | None
    duplicate_candidates: list[DuplicateCandidate]
    missing_information: list[MissingInformationItem]
    follow_up_draft: FollowUpDraft | None
    narrative_draft: NarrativeDraft | None

    # Evaluation
    evaluation_results: Annotated[list[EvaluationResult], operator.add]
    retry_counts: dict[str, int]

    # Human review
    review_status: ReviewStatus | None
    reviewer_changes: Annotated[list[ReviewerChange], operator.add]

    # Observability / traceability
    agent_events: Annotated[list[AgentEvent], operator.add]
    tool_events: Annotated[list[ToolEvent], operator.add]
    memory_events: Annotated[list[MemoryEvent], operator.add]
    retrieval_events: Annotated[list[RetrievalEvent], operator.add]
    errors: Annotated[list[ErrorRecord], operator.add]

    # Model / cost bookkeeping
    model_version: str
    prompt_versions: dict[str, str]
    total_cost: float


def new_case_id() -> str:
    return f"case_{uuid.uuid4().hex[:12]}"


def new_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex[:12]}"


def create_initial_state(
    email: EmailMessage,
    attachments: list[Attachment] | None = None,
    case_id: str | None = None,
    trace_id: str | None = None,
    model_version: str = "phase1-scaffold",
) -> CaseState:
    """Build the starting CaseState for a new case (before Goal Manager runs)."""
    resolved_case_id = case_id or new_case_id()
    return CaseState(
        goal=None,
        success_criteria=[],
        goal_status=GoalStatus.CREATED,
        execution_plan=None,
        plan_history=[],
        current_step=None,
        case_id=resolved_case_id,
        trace_id=trace_id or new_trace_id(),
        email=email,
        attachments=attachments or [],
        source_passages=[],
        extracted_fields=None,
        minimum_criteria=None,
        seriousness_triage=None,
        duplicate_candidates=[],
        missing_information=[],
        follow_up_draft=None,
        narrative_draft=None,
        evaluation_results=[],
        retry_counts={},
        review_status=None,
        reviewer_changes=[],
        agent_events=[],
        tool_events=[],
        memory_events=[],
        retrieval_events=[],
        errors=[],
        model_version=model_version,
        prompt_versions={},
        total_cost=0.0,
    )


def utcnow() -> datetime:
    return datetime.now(UTC)
