"""Unit tests for the typed Pydantic domain models in src/models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import (
    AgentDecision,
    AgentName,
    CaseGoal,
    Citation,
    DecisionOutcome,
    DuplicateCandidate,
    EvaluationCheck,
    EvaluationResult,
    ExecutionPlan,
    FieldValue,
    FollowUpDraft,
    GoalStatus,
    MinimumCriteriaResult,
    MissingInformationItem,
    PlanStep,
    ReportSection,
    ReviewDecision,
    ReviewStatus,
    SuccessCriterion,
    TriageFinding,
    TriageIndicator,
    TriageReport,
    TriageResult,
)


def test_case_goal_defaults_and_boundaries_present() -> None:
    goal = CaseGoal(case_id="case_001")
    assert goal.status == GoalStatus.CREATED
    assert goal.success_criteria == []
    assert any("human approval" in b for b in goal.boundaries)
    assert not goal.is_met()


def test_case_goal_is_met_only_when_all_criteria_met() -> None:
    goal = CaseGoal(
        case_id="case_001",
        success_criteria=[
            SuccessCriterion(criterion_id="c1", description="extract fields", met=True),
            SuccessCriterion(criterion_id="c2", description="human review done", met=False),
        ],
    )
    assert not goal.is_met()
    goal.success_criteria[1].met = True
    assert goal.is_met()


def test_execution_plan_requires_steps_typed() -> None:
    plan = ExecutionPlan(
        plan_id="plan_001",
        case_id="case_001",
        steps=[
            PlanStep(
                step_id="s1", agent=AgentName.SUBJECT_READER, description="read email subject"
            ),
            PlanStep(
                step_id="s2",
                agent=AgentName.ATTACHMENT_READER,
                description="skip: no attachments",
                skip=True,
                skip_reason="no attachments present",
            ),
        ],
    )
    assert plan.version == 1
    assert plan.steps[1].skip
    assert plan.steps[1].skip_reason


def test_agent_decision_confidence_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        AgentDecision(
            agent=AgentName.MEDICAL_EXTRACTION,
            case_id="case_001",
            decision="meets_minimum_criteria",
            confidence=1.5,  # out of [0, 1]
            decision_summary="test",
            next_action=DecisionOutcome.PROCEED,
            requires_human_review=False,
        )


def test_agent_decision_valid_construction() -> None:
    decision = AgentDecision(
        agent=AgentName.TRIAGE,
        case_id="case_001",
        decision="hospitalization_indicator_present",
        evidence=["passage_003"],
        confidence=0.9,
        decision_summary="Explicit hospitalization mention found in discharge summary.",
        next_action=DecisionOutcome.ESCALATE_TO_HUMAN,
        requires_human_review=True,
    )
    assert decision.requires_human_review is True
    assert decision.next_action == DecisionOutcome.ESCALATE_TO_HUMAN


def test_field_value_carries_citation_and_confidence() -> None:
    field = FieldValue(
        field_name="product.product_name",
        value="DemoInsulex",
        citations=[Citation(passage_id="p1", quoted_text="DemoInsulex", page_number=1)],
        confidence=0.95,
    )
    assert field.citations[0].passage_id == "p1"
    assert field.conflict_status.value == "none"


def test_minimum_criteria_result_property() -> None:
    complete = MinimumCriteriaResult(
        has_identifiable_patient=True,
        has_identifiable_reporter=True,
        has_suspect_product=True,
        has_adverse_event=True,
    )
    assert complete.meets_minimum_criteria

    incomplete = MinimumCriteriaResult(
        has_identifiable_patient=True,
        has_identifiable_reporter=True,
        has_suspect_product=False,
        has_adverse_event=True,
        missing_criteria=["suspect_product"],
    )
    assert not incomplete.meets_minimum_criteria
    assert "suspect_product" in incomplete.missing_criteria


def test_triage_result_any_indicator_present() -> None:
    empty = TriageResult()
    assert not empty.any_indicator_present

    with_finding = TriageResult(
        findings=[TriageFinding(indicator=TriageIndicator.HOSPITALIZATION)]
    )
    assert with_finding.any_indicator_present
    assert with_finding.requires_human_confirmation


def test_duplicate_candidate_scores() -> None:
    candidate = DuplicateCandidate(
        candidate_case_id="case_099",
        bm25_score=0.8,
        vector_score=0.7,
        combined_score=0.75,
        matching_fields=["product.product_name"],
    )
    assert candidate.combined_score == 0.75


def test_missing_information_and_follow_up_draft() -> None:
    item = MissingInformationItem(
        field_name="product.dose",
        reason="missing",
        follow_up_question="What was the dose administered?",
    )
    draft = FollowUpDraft(
        draft_id="fu_001",
        case_id="case_001",
        subject="Follow-up needed on your report",
        body="Could you confirm the dose and treatment start date?",
        questions=[item],
    )
    assert draft.editable
    assert draft.questions[0].reason == "missing"


def test_triage_report_full_text() -> None:
    report = TriageReport(
        report_id="n_001",
        case_id="case_001",
        sections=[
            ReportSection(text="Patient received DemoInsulex."),
            ReportSection(text="Patient was hospitalized three days later."),
        ],
    )
    expected = "Patient received DemoInsulex. Patient was hospitalized three days later."
    assert report.full_text == expected


def test_evaluation_result_passed_property() -> None:
    passing = EvaluationResult(
        evaluation_id="eval_001",
        case_id="case_001",
        step_name="medical_extraction",
        checks=[EvaluationCheck(name="schema_valid", passed=True)],
    )
    assert passing.passed

    failing = EvaluationResult(
        evaluation_id="eval_002",
        case_id="case_001",
        step_name="narrative",
        checks=[EvaluationCheck(name="citation_coverage", passed=False, detail="uncited sentence")],
        unsupported_claims=["sentence 2 has no citation"],
    )
    assert not failing.passed


def test_review_status_defaults_to_pending() -> None:
    review = ReviewStatus(case_id="case_001")
    assert review.decision == ReviewDecision.PENDING
    assert review.reviewer_id is None
