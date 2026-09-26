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
from src.ui import case_workspace

PLACEHOLDER_PAGES = [
    "Dashboard",
    "New Case Intake",
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
    st.caption(
        f"LLM provider: {settings.llm_provider.value} · DB backend: {settings.db_backend.value}"
    )

    page = st.sidebar.radio("Pages", ["Case Workspace", *PLACEHOLDER_PAGES])

    if page == "Case Workspace":
        case_workspace.render()
    else:
        st.subheader(page)
        st.info(f"'{page}' is not yet implemented in this prototype. See progress.md.")


if __name__ == "__main__":
    main()
