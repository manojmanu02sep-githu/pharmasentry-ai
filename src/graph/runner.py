"""LangGraph wiring for the 12-node case pipeline.

Node order and edges follow CLAUDE.md's Orchestration section and
`config/config.yaml`'s `graph.node_order` exactly. A `MemorySaver`
checkpointer persists working memory per `case_id` (LangGraph thread),
satisfying the "typed state and checkpointing" requirement -- this is
process-local (in-memory) for this prototype, matching the same
in-memory posture as `InMemoryCaseStateStore` (Phase 4/8 note in
`src/tools/workflow.py` applies here too).

Human review is a hard stop: the graph ends after `human_review` runs
and sets `goal_status=AWAITING_HUMAN`. Resuming after a real reviewer
decision is a separate, explicit call (`src.agents.human_review.apply_review_decision`)
made by the UI layer -- never an automatic continuation of this graph.
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents import (
    duplicate_search,
    email_body_reader,
    evaluator,
    goal_agent,
    human_review,
    medical_extraction,
    pdf_docx_reader,
    planner,
    report_generator,
    subject_reader,
    supervisor,
    triage,
)
from src.graph.routing import (
    CONTINUE,
    HUMAN_REVIEW,
    after_email_body_reader,
    after_pdf_docx_reader,
    after_subject_reader,
)
from src.graph.state import CaseState

NodeFn = Callable[[dict], dict]

NODES: tuple[tuple[str, NodeFn], ...] = (
    ("goal_agent", goal_agent.run),
    ("planner", planner.run),
    ("supervisor", supervisor.run),
    ("subject_reader", subject_reader.run),
    ("email_body_reader", email_body_reader.run),
    ("pdf_docx_reader", pdf_docx_reader.run),
    ("medical_extraction", medical_extraction.run),
    ("triage", triage.run),
    ("duplicate_search", duplicate_search.run),
    ("report_generator", report_generator.run),
    ("evaluator", evaluator.run),
    ("human_review", human_review.run),
)


def build_graph() -> StateGraph:
    graph = StateGraph(CaseState)
    for name, fn in NODES:
        # LangGraph's stubs require each node's Callable to be parameterized
        # on the exact graph state type; our node functions intentionally
        # take/return plain dict (partial-state updates) rather than the
        # CaseState TypedDict, matching LangGraph's actual runtime contract.
        graph.add_node(name, fn)  # type: ignore[call-overload]

    graph.add_edge(START, "goal_agent")
    graph.add_edge("goal_agent", "planner")
    graph.add_edge("planner", "supervisor")
    graph.add_edge("supervisor", "subject_reader")

    graph.add_conditional_edges(
        "subject_reader",
        after_subject_reader,
        {CONTINUE: "email_body_reader", HUMAN_REVIEW: "human_review"},
    )
    graph.add_conditional_edges(
        "email_body_reader",
        after_email_body_reader,
        {CONTINUE: "pdf_docx_reader", HUMAN_REVIEW: "human_review"},
    )
    graph.add_conditional_edges(
        "pdf_docx_reader",
        after_pdf_docx_reader,
        {CONTINUE: "medical_extraction", HUMAN_REVIEW: "human_review"},
    )

    graph.add_edge("medical_extraction", "triage")
    graph.add_edge("triage", "duplicate_search")
    graph.add_edge("duplicate_search", "report_generator")
    graph.add_edge("report_generator", "evaluator")
    graph.add_edge("evaluator", "human_review")
    graph.add_edge("human_review", END)

    return graph


def compiled_app():
    return build_graph().compile(checkpointer=MemorySaver())


def run_case(initial_state: CaseState):
    """Run one case to completion (through human_review) and return the
    final CaseState. Uses `case_id` as the LangGraph thread id so the
    checkpointer keeps each case's working memory isolated."""
    app = compiled_app()
    config = {"configurable": {"thread_id": initial_state["case_id"]}}
    return app.invoke(initial_state, config=config)
