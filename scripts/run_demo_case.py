"""Run CLAUDE.md's synthetic demo case through the full 12-node graph and
print a summary of the resulting CaseState. For manual verification --
never used as a source of truth for automated tests (see
tests/integration/test_demo_case_e2e.py for that)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.demo_case import build_demo_case_initial_state  # noqa: E402
from src.graph.runner import run_case  # noqa: E402


def main() -> None:
    initial_state = build_demo_case_initial_state()
    final_state = run_case(initial_state)

    print(f"case_id: {final_state['case_id']}")
    print(f"trace_id: {final_state['trace_id']}")
    print(f"goal_status: {final_state['goal_status'].value}")
    print(f"current_step: {final_state['current_step']}")
    print(f"agent_events: {len(final_state['agent_events'])}")
    print(f"tool_events: {len(final_state['tool_events'])}")
    print(f"memory_events: {len(final_state['memory_events'])}")
    print(f"retrieval_events: {len(final_state['retrieval_events'])}")
    print(f"errors: {len(final_state['errors'])}")
    print(f"source_passages: {len(final_state['source_passages'])}")

    fields = final_state.get("extracted_fields")
    if fields:
        print("\nExtracted fields:")
        for fv in (
            fields.patient.age,
            fields.patient.sex,
            fields.reporter.name,
            fields.reporter.reporter_type,
            fields.product.product_name,
            fields.product.dose,
            fields.event.event_description,
            fields.treatment.action_taken,
            fields.outcome.outcome_description,
            fields.outcome.hospitalized,
        ):
            print(
                f"  {fv.field_name}: value={fv.value!r} confidence={fv.confidence} "
                f"citations={len(fv.citations)} conflict={fv.conflict_status.value}"
            )

    minimum_criteria = final_state.get("minimum_criteria")
    if minimum_criteria:
        print(f"\nminimum_criteria_met: {minimum_criteria.meets_minimum_criteria}")
        print(f"missing_criteria: {minimum_criteria.missing_criteria}")

    triage_result = final_state.get("triage_result")
    if triage_result:
        print(f"\ntriage findings: {len(triage_result.findings)}")
        for f in triage_result.findings:
            print(f"  indicator={f.indicator} citations={len(f.citations)}")

    print(f"\nduplicate_candidates: {len(final_state.get('duplicate_candidates') or [])}")
    print(f"missing_information: {len(final_state.get('missing_information') or [])}")
    for m in final_state.get("missing_information") or []:
        print(f"  {m.field_name}: {m.follow_up_question}")

    triage_report = final_state.get("triage_report")
    if triage_report:
        print(f"\nnarrative sections: {len(triage_report.sections)}")
        for s in triage_report.sections:
            print(f"  - {s.text} [{len(s.citations)} citation(s)]")
        print(f"unsupported_claim_flags: {triage_report.unsupported_claim_flags}")

    evaluation_results = final_state.get("evaluation_results") or []
    if evaluation_results:
        result = evaluation_results[-1]
        print(f"\nevaluation: passed={result.passed} evidence_coverage={result.evidence_coverage}")
        for check in result.checks:
            print(f"  [{('PASS' if check.passed else 'FAIL')}] {check.name}: {check.detail}")

    review_status = final_state.get("review_status")
    if review_status:
        print(f"\nreview_status: decision={review_status.decision.value}")


if __name__ == "__main__":
    main()
