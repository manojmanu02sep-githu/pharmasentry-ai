"""Case Workspace page: run the synthetic demo case end-to-end and let an
authorized reviewer see evidence beside extracted fields (value, source,
citation, confidence, conflict status) and approve/reject/request changes.
Never auto-advances a case past human review."""

from __future__ import annotations

import streamlit as st

from src.agents.human_review import apply_review_decision
from src.graph.demo_case import build_demo_case_initial_state
from src.graph.runner import run_case
from src.models.enums import ReviewDecision

STATE_KEY = "case_workspace_state"

_FIELD_LABELS = [
    ("patient.age", lambda f: f.patient.age),
    ("patient.sex", lambda f: f.patient.sex),
    ("reporter.name", lambda f: f.reporter.name),
    ("reporter.reporter_type", lambda f: f.reporter.reporter_type),
    ("product.product_name", lambda f: f.product.product_name),
    ("product.dose", lambda f: f.product.dose),
    ("product.treatment_start_date", lambda f: f.product.treatment_start_date),
    ("event.event_description", lambda f: f.event.event_description),
    ("event.event_onset_date", lambda f: f.event.event_onset_date),
    ("treatment.action_taken", lambda f: f.treatment.action_taken),
    ("outcome.outcome_description", lambda f: f.outcome.outcome_description),
    ("outcome.hospitalized", lambda f: f.outcome.hospitalized),
]


def _render_evidence(state: dict) -> None:
    email = state.get("email")
    st.subheader("Original evidence")
    if email:
        st.markdown(f"**Email** — from `{email.sender}` — subject: *{email.subject}*")
        st.text_area("Email body", email.body, height=120, disabled=True, key="email_body_ro")
    for attachment in state.get("attachments") or []:
        st.markdown(
            f"**Attachment** — `{attachment.filename}` ({attachment.validation_status.value})"
        )
    for passage in state.get("source_passages") or []:
        with st.expander(f"Passage `{passage.passage_id}` ({passage.source_type})"):
            st.text(passage.text)


def _render_fields(state: dict) -> None:
    fields = state.get("extracted_fields")
    st.subheader("Extracted fields")
    if fields is None:
        st.info("No extracted fields yet.")
        return
    rows = []
    for name, getter in _FIELD_LABELS:
        fv = getter(fields)
        rows.append(
            {
                "field": name,
                "value": fv.value,
                "confidence": fv.confidence,
                "citations": len(fv.citations),
                "conflict_status": fv.conflict_status.value,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_minimum_criteria(state: dict) -> None:
    mc = state.get("minimum_criteria")
    if not mc:
        return
    st.subheader("Minimum case criteria")
    st.write(f"Met: **{mc.meets_minimum_criteria}** — missing: {mc.missing_criteria or 'none'}")


def _render_triage(state: dict) -> None:
    triage = state.get("triage_result")
    if not triage:
        return
    st.subheader("Seriousness triage (AI-suggested — requires human confirmation)")
    if not triage.findings:
        st.write("No explicit seriousness indicators detected.")
    for f in triage.findings:
        st.write(f"- {f.indicator.value} ({len(f.citations)} citation(s))")
    st.caption(f"AI-suggested priority: {triage.suggested_priority.value} — {triage.rationale}")


def _render_duplicates(state: dict) -> None:
    candidates = state.get("duplicate_candidates") or []
    st.subheader("Top potential duplicates (never auto-merged)")
    if not candidates:
        st.write("No candidates found.")
        return
    for c in candidates:
        st.write(
            f"- `{c.candidate_case_id}` combined_score={c.combined_score:.3f} "
            f"matching={c.matching_fields} conflicting={c.conflicting_fields}"
        )


def _render_missing_and_narrative(state: dict) -> None:
    missing = state.get("missing_information") or []
    st.subheader("Missing information")
    if not missing:
        st.write("None.")
    for m in missing:
        st.write(f"- **{m.field_name}**: {m.follow_up_question}")

    follow_up = state.get("follow_up_draft")
    if follow_up:
        st.subheader("Editable follow-up draft")
        st.text_area("Subject", follow_up.subject, key="fu_subject", disabled=True)
        st.text_area("Body", follow_up.body, height=140, key="fu_body", disabled=True)

    report = state.get("triage_report")
    if report:
        st.subheader("Source-cited narrative draft")
        for s in report.sections:
            st.write(f"- {s.text} ({len(s.citations)} citation(s))")
        if report.unsupported_claim_flags:
            st.error(f"Unsupported claims flagged: {report.unsupported_claim_flags}")


def _render_evaluation(state: dict) -> None:
    results = state.get("evaluation_results") or []
    if not results:
        return
    result = results[-1]
    st.subheader("Evaluation")
    st.write(f"Passed: **{result.passed}** — evidence coverage: {result.evidence_coverage:.2f}")
    for check in result.checks:
        icon = "✅" if check.passed else "❌"
        st.write(f"{icon} {check.name}: {check.detail}")


def _render_review_controls(state: dict) -> None:
    st.subheader("Human review")
    review_status = state.get("review_status")
    if review_status:
        st.write(f"Current decision: **{review_status.decision.value}**")
    if state.get("goal_status") is None or state["goal_status"].value != "awaiting_human":
        st.info("Case is not awaiting human review.")
        return

    reviewer_id = st.text_input("Reviewer ID", value="reviewer_demo", key="reviewer_id")
    notes = st.text_area("Notes", key="review_notes")
    col1, col2, col3 = st.columns(3)
    decision = None
    if col1.button("Approve", type="primary"):
        decision = ReviewDecision.APPROVED
    if col2.button("Reject"):
        decision = ReviewDecision.REJECTED
    if col3.button("Request changes"):
        decision = ReviewDecision.CHANGES_REQUESTED

    if decision is not None:
        update = apply_review_decision(
            state,
            decision=decision,
            reviewer_id=reviewer_id,
            notes=notes or None,
            field_edits=[],
        )
        state.update(update)
        st.session_state[STATE_KEY] = state
        st.success(f"Recorded reviewer decision: {decision.value}. No automated action taken.")
        st.rerun()


def render() -> None:
    st.header("Case Workspace")
    st.caption(
        "Runs CLAUDE.md's synthetic demo case (fictional product DemoGlutide) "
        "through the full 12-node pipeline and pauses for human review."
    )

    if st.button("Run demo case"):
        with st.spinner("Running case through the agent pipeline..."):
            initial_state = build_demo_case_initial_state()
            final_state = run_case(initial_state)
        st.session_state[STATE_KEY] = dict(final_state)

    state = st.session_state.get(STATE_KEY)
    if state is None:
        st.info("Click 'Run demo case' to process the synthetic demo email and attachment.")
        return

    st.success(
        f"case_id={state['case_id']} · goal_status={state['goal_status'].value} · "
        f"agent_events={len(state['agent_events'])} · errors={len(state['errors'])}"
    )

    evidence_col, fields_col = st.columns(2)
    with evidence_col:
        _render_evidence(state)
    with fields_col:
        _render_fields(state)
        _render_minimum_criteria(state)
        _render_triage(state)

    _render_duplicates(state)
    _render_missing_and_narrative(state)
    _render_evaluation(state)
    _render_review_controls(state)
