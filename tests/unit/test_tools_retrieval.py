"""Unit tests for src/tools/retrieval.py."""

from __future__ import annotations

from src.models.enums import AgentName
from src.tools.base import ToolContext
from src.tools.retrieval import (
    Bm25SearchInput,
    Bm25SearchTool,
    CorpusDocument,
    ExactCaseSearchInput,
    ExactCaseSearchTool,
    MetadataFilterInput,
    MetadataFilterTool,
    ReciprocalRankFusionInput,
    ReciprocalRankFusionTool,
    RetrieveCaseEvidenceInput,
    RetrieveCaseEvidenceTool,
    VectorSearchInput,
    VectorSearchTool,
    deterministic_embedding,
)

CTX = ToolContext(
    case_id="case_001",
    trace_id="trace_001",
    agent=AgentName.DUPLICATE_SEARCH,
    authorized_tools=frozenset(
        {
            "exact_case_search", "bm25_search", "vector_search", "metadata_filter",
            "reciprocal_rank_fusion", "retrieve_case_evidence",
        }
    ),
)

CORPUS = [
    CorpusDocument(
        case_id="case_a",
        text="Patient took DemoGluca and developed severe abdominal pain and vomiting.",
        fields={"product.product_name": "DemoGluca", "event.event_description": "abdominal pain"},
    ),
    CorpusDocument(
        case_id="case_b",
        text="Patient reports mild headache after taking DemoCardolol.",
        fields={"product.product_name": "DemoCardolol", "event.event_description": "headache"},
    ),
    CorpusDocument(
        case_id="case_c",
        text="Another severe abdominal pain and vomiting case involving DemoGluca.",
        fields={"product.product_name": "DemoGluca", "event.event_description": "abdominal pain"},
    ),
]


def test_exact_case_search_matches_on_all_query_fields() -> None:
    output, _ = ExactCaseSearchTool().run(
        ExactCaseSearchInput(
            query_fields={"product.product_name": "DemoGluca"}, corpus=CORPUS
        ),
        CTX,
    )
    assert set(output.matching_case_ids) == {"case_a", "case_c"}


def test_exact_case_search_no_match() -> None:
    output, _ = ExactCaseSearchTool().run(
        ExactCaseSearchInput(query_fields={"product.product_name": "DemoZanix"}, corpus=CORPUS),
        CTX,
    )
    assert output.matching_case_ids == []


def test_bm25_search_ranks_lexically_similar_case_highest() -> None:
    output, _ = Bm25SearchTool().run(
        Bm25SearchInput(
            query_text="severe abdominal pain and vomiting DemoGluca", corpus=CORPUS, top_k=3
        ),
        CTX,
    )
    top_ids = [r.case_id for r in output.results]
    assert top_ids[0] in {"case_a", "case_c"}
    assert "case_b" in top_ids  # present, just ranked lower
    assert output.results[0].score >= output.results[-1].score


def test_deterministic_embedding_is_reproducible() -> None:
    v1 = deterministic_embedding("severe abdominal pain")
    v2 = deterministic_embedding("severe abdominal pain")
    assert v1 == v2


def test_vector_search_ranks_semantically_similar_text_highest() -> None:
    output, _ = VectorSearchTool().run(
        VectorSearchInput(query_text="severe abdominal pain and vomiting", corpus=CORPUS, top_k=3),
        CTX,
    )
    assert output.results[0].case_id in {"case_a", "case_c"}


def test_metadata_filter_narrows_to_matching_field() -> None:
    output, _ = MetadataFilterTool().run(
        MetadataFilterInput(
            candidate_case_ids=["case_a", "case_b", "case_c"],
            corpus=CORPUS,
            filters={"product.product_name": "DemoGluca"},
        ),
        CTX,
    )
    assert set(output.filtered_case_ids) == {"case_a", "case_c"}


def test_reciprocal_rank_fusion_combines_two_rankings() -> None:
    output, _ = ReciprocalRankFusionTool().run(
        ReciprocalRankFusionInput(
            rankings=[["case_a", "case_b", "case_c"], ["case_c", "case_a", "case_b"]],
            k=60,
            top_k=3,
        ),
        CTX,
    )
    # case_a and case_c both appear near the top of both rankings -> highest fused score
    top_ids = {r.case_id for r in output.results[:2]}
    assert top_ids == {"case_a", "case_c"}


def test_reciprocal_rank_fusion_never_merges_just_ranks() -> None:
    output, _ = ReciprocalRankFusionTool().run(
        ReciprocalRankFusionInput(rankings=[["x", "y"], ["y", "z"]], k=60, top_k=5), CTX
    )
    ids = [r.case_id for r in output.results]
    assert set(ids) == {"x", "y", "z"}  # nothing merged/dropped incorrectly


def test_retrieve_case_evidence_reports_matching_and_conflicting_fields() -> None:
    candidate = CorpusDocument(
        case_id="case_a",
        text="Some evidence text.",
        fields={"product.product_name": "DemoGluca", "product.dose": "10mg"},
    )
    output, _ = RetrieveCaseEvidenceTool().run(
        RetrieveCaseEvidenceInput(
            query_fields={"product.product_name": "DemoGluca", "product.dose": "50mg"},
            candidate=candidate,
        ),
        CTX,
    )
    assert output.matching_fields == ["product.product_name"]
    assert output.conflicting_fields == ["product.dose"]
    assert output.evidence_snippets
