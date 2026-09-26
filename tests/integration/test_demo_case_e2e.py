"""End-to-end integration test: CLAUDE.md's synthetic demo case must run
through the full 12-node graph and reach human review with real evidence,
extracted fields, and an evaluation result -- never a fabricated or
skipped outcome."""

from __future__ import annotations

from src.graph.demo_case import build_demo_case_initial_state
from src.graph.runner import run_case
from src.models.enums import GoalStatus


def test_demo_case_reaches_human_review_with_real_evidence() -> None:
    initial_state = build_demo_case_initial_state()
    final_state = run_case(initial_state)

    assert final_state["goal_status"] == GoalStatus.AWAITING_HUMAN
    assert final_state["current_step"] == "human_review"
    assert final_state["errors"] == []

    assert final_state["source_passages"], "expected non-empty evidence passages"

    fields = final_state["extracted_fields"]
    assert fields is not None
    assert fields.product.product_name.value is not None
    assert fields.product.product_name.citations, "product name must be grounded in evidence"
    assert fields.outcome.hospitalized.value is not None
    assert fields.outcome.hospitalized.citations, "hospitalized flag must be grounded in evidence"

    minimum_criteria = final_state["minimum_criteria"]
    assert minimum_criteria is not None
    assert minimum_criteria.meets_minimum_criteria is True

    evaluation_results = final_state["evaluation_results"]
    assert evaluation_results, "expected a real EvaluationResult, not a skipped step"
    result = evaluation_results[-1]
    assert result.passed is True
    assert result.evidence_coverage == 1.0

    review_status = final_state["review_status"]
    assert review_status is not None
    assert review_status.decision.value == "pending"
