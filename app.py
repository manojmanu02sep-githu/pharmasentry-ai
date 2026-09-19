"""PharmaSentry AI — Streamlit entry point.

Phase 1 scaffold: the full multi-page UI (Dashboard, New Case Intake, Case
Workspace, Planning and Agent Progress, Duplicate Review, Human Review
Queue, Evaluation, Observability and Traceability, System Configuration,
About and Limitations) is implemented in a later phase — see progress.md.
Run with: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from config.settings import get_settings

PHASE_1_PAGES = [
    "Dashboard",
    "New Case Intake",
    "Case Workspace",
    "Planning and Agent Progress",
    "Duplicate Review",
    "Human Review Queue",
    "Evaluation",
    "Observability and Traceability",
    "System Configuration",
    "About and Limitations",
]


def main() -> None:
    st.set_page_config(page_title="PharmaSentry AI (prototype)", layout="wide")
    settings = get_settings()

    st.title("PharmaSentry AI — Educational Prototype")
    st.warning(
        "This is an educational prototype using only synthetic data. It is "
        "not a medical device, safety database, regulatory submission "
        "system, or clinical decision system. No output here is a final "
        "medical or regulatory decision."
    )
    st.caption(f"LLM provider: {settings.llm_provider.value} · "
               f"DB backend: {settings.db_backend.value}")

    st.subheader("Build status")
    st.write(
        "The full UI is built in later phases. Planned pages, in order:"
    )
    for page in PHASE_1_PAGES:
        st.markdown(f"- {page}")

    st.info("See progress.md in the repository root for phase-by-phase status.")


if __name__ == "__main__":
    main()
