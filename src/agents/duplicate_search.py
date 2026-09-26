"""Duplicate Agent: hybrid BM25 + vector retrieval over the synthetic
case corpus via `DuplicateSearchService`. Never receives `ReporterInfo`
or any reporter contact details -- only product/event/patient fields are
built into `query_fields`, and the corpus itself already excludes
`reporter.name` (Context Isolation and Delegation). Never merges cases or
makes a final duplicate decision."""

from __future__ import annotations

import time

from config.settings import get_settings
from src.agents._common import make_agent_event, make_tool_context
from src.models.audit import RetrievalEvent
from src.models.enums import AgentName, DecisionOutcome
from src.models.reasoning import AgentDecision
from src.retrieval.corpus import build_corpus_from_golden_dataset
from src.retrieval.duplicate_search import DuplicateSearchService
from src.retrieval.hybrid_ranker import HybridRanker

NODE = AgentName.DUPLICATE_SEARCH


def run(state: dict) -> dict:
    started = time.monotonic()
    case_id = state["case_id"]
    trace_id = state["trace_id"]
    ctx = make_tool_context(case_id, trace_id, NODE)

    fields = state.get("extracted_fields")
    query_fields: dict[str, str | None] = {}
    query_text_parts: list[str] = []
    if fields is not None:
        query_fields = {
            "product.product_name": fields.product.product_name.value,
            "product.dose": fields.product.dose.value,
            "event.event_description": fields.event.event_description.value,
            "outcome.hospitalized": fields.outcome.hospitalized.value,
        }
        query_text_parts = [v for v in query_fields.values() if v]
    email_body = state["email"].body if state.get("email") else ""
    query_text = " ".join(query_text_parts) or email_body

    corpus = build_corpus_from_golden_dataset()
    ranker, ranker_memory_event = HybridRanker.from_config(ctx)
    service = DuplicateSearchService.from_settings(get_settings())
    candidates, events = service.search(query_text, query_fields, corpus, ranker, ctx)

    tool_events = [e for e in events if not isinstance(e, RetrievalEvent)]
    retrieval_events = [e for e in events if isinstance(e, RetrievalEvent)]

    decision = AgentDecision(
        agent=NODE,
        case_id=case_id,
        decision="duplicates_found" if candidates else "no_duplicates_found",
        evidence=[c.candidate_case_id for c in candidates[:5]],
        confidence=0.7,
        decision_summary=f"Hybrid retrieval over {len(corpus)} synthetic case(s) returned "
        f"{len(candidates)} candidate(s); no auto-merge performed.",
        next_action=DecisionOutcome.ESCALATE_TO_HUMAN if candidates else DecisionOutcome.PROCEED,
        requires_human_review=bool(candidates),
    )
    latency_ms = (time.monotonic() - started) * 1000
    agent_event = make_agent_event(
        decision,
        trace_id,
        node="duplicate_search",
        model_version=state["model_version"],
        latency_ms=latency_ms,
    )

    return {
        "duplicate_candidates": candidates,
        "current_step": "duplicate_search",
        "agent_events": [agent_event],
        "tool_events": tool_events,
        "retrieval_events": retrieval_events,
        "memory_events": [ranker_memory_event],
    }
