"""Unit tests for src/retrieval/duplicate_search.py."""

from __future__ import annotations

from src.models.audit import RetrievalEvent
from src.models.enums import AgentName
from src.retrieval.duplicate_search import DuplicateSearchService
from src.retrieval.hybrid_ranker import HybridRanker
from src.tools.base import ToolContext
from src.tools.retrieval import CorpusDocument

CTX = ToolContext(
    case_id="case_a",
    trace_id="trace_001",
    agent=AgentName.DUPLICATE_SEARCH,
    authorized_tools=frozenset(
        {"bm25_search", "vector_search", "retrieve_case_evidence"}
    ),
)

CORPUS = [
    CorpusDocument(
        case_id="case_a",
        text="Patient took DemoGlutide and developed severe abdominal pain and vomiting.",
        fields={"product.product_name": "DemoGlutide", "event.event_description": "abdominal pain"},
    ),
    CorpusDocument(
        case_id="case_b",
        text="Patient reports mild headache after taking DemoBasalin.",
        fields={"product.product_name": "DemoBasalin", "event.event_description": "headache"},
    ),
    CorpusDocument(
        case_id="case_c",
        text="Another severe abdominal pain and vomiting case involving DemoGlutide.",
        fields={"product.product_name": "DemoGlutide", "event.event_description": "abdominal pain"},
    ),
]


def test_search_returns_candidates_never_merges() -> None:
    service = DuplicateSearchService()
    ranker = HybridRanker(bm25_weight=0.5, vector_weight=0.5, top_k=2)

    candidates, events = service.search(
        query_text="severe abdominal pain and vomiting DemoGlutide",
        query_fields={
            "product.product_name": "DemoGlutide",
            "event.event_description": "abdominal pain",
        },
        corpus=CORPUS,
        ranker=ranker,
        ctx=CTX,
    )

    assert {c.candidate_case_id for c in candidates} == {"case_a", "case_c"}
    assert "product.product_name" in candidates[0].matching_fields
    assert candidates[0].evidence_snippets

    # Candidate output has no fields that could constitute an auto-merge or
    # final decision — only scores, matches, conflicts, and evidence.
    assert not hasattr(candidates[0], "merged")
    assert any(isinstance(e, RetrievalEvent) for e in events)


def test_search_reports_conflicting_fields() -> None:
    service = DuplicateSearchService()
    ranker = HybridRanker(bm25_weight=0.5, vector_weight=0.5, top_k=3)

    candidates, _ = service.search(
        query_text="severe abdominal pain and vomiting DemoGlutide",
        query_fields={"product.product_name": "DemoGlutide", "event.event_description": "headache"},
        corpus=CORPUS,
        ranker=ranker,
        ctx=CTX,
    )

    top = next(c for c in candidates if c.candidate_case_id == "case_c")
    assert "event.event_description" in top.conflicting_fields
