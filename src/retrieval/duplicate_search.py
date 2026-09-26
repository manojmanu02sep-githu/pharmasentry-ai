"""Duplicate Search: hybrid BM25 + vector retrieval producing
`DuplicateCandidate` objects for human review.

This service NEVER merges cases and never makes a final duplicate
decision (CLAUDE.md Hybrid RAG: "RAG must not make the final duplicate or
regulatory decision"). It also never sees reporter contact details: the
corpus it searches (`src.retrieval.corpus`) excludes `reporter.name`
before this code ever runs (Context Isolation and Delegation).
"""

from __future__ import annotations

import logging

from config.settings import Settings
from src.models.audit import RetrievalEvent, ToolEvent
from src.models.duplicates import DuplicateCandidate
from src.retrieval.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
    get_embedding_provider,
)
from src.retrieval.hybrid_ranker import HybridRanker
from src.tools.base import ToolContext
from src.tools.retrieval import (
    Bm25SearchInput,
    Bm25SearchTool,
    CorpusDocument,
    RetrieveCaseEvidenceInput,
    RetrieveCaseEvidenceTool,
    VectorSearchInput,
    VectorSearchTool,
)

logger = logging.getLogger(__name__)


class DuplicateSearchService:
    def __init__(self, embedding_provider: EmbeddingProvider | None = None) -> None:
        self._embedding_provider = embedding_provider or DeterministicEmbeddingProvider()
        self._bm25_tool = Bm25SearchTool()
        self._vector_tool = VectorSearchTool(embed_fn=self._embedding_provider.embed)
        self._evidence_tool = RetrieveCaseEvidenceTool()

    @classmethod
    def from_settings(cls, settings: Settings) -> DuplicateSearchService:
        """Build a service using the embedding provider configured in
        Settings (`EMBEDDING_PROVIDER`/`EMBEDDING_MODEL`) — for real
        callers. Tests that don't care about the provider should keep
        using the bare constructor, which defaults to deterministic."""
        provider = get_embedding_provider(
            settings.embedding_provider.value, settings.embedding_model
        )
        return cls(embedding_provider=provider)

    def search(
        self,
        query_text: str,
        query_fields: dict[str, str | None],
        corpus: list[CorpusDocument],
        ranker: HybridRanker,
        ctx: ToolContext,
    ) -> tuple[list[DuplicateCandidate], list[ToolEvent | RetrievalEvent]]:
        events: list[ToolEvent | RetrievalEvent] = []
        fusion_breadth = max(ranker.top_k, 10)

        bm25_output, bm25_event = self._bm25_tool.run(
            Bm25SearchInput(query_text=query_text, corpus=corpus, top_k=fusion_breadth), ctx
        )
        events.append(bm25_event)
        vector_output, vector_event = self._vector_tool.run(
            VectorSearchInput(query_text=query_text, corpus=corpus, top_k=fusion_breadth), ctx
        )
        events.append(vector_event)

        bm25_scores = {r.case_id: r.score for r in bm25_output.results}
        vector_scores = {r.case_id: r.score for r in vector_output.results}
        combined = ranker.combine(
            [r.case_id for r in bm25_output.results], [r.case_id for r in vector_output.results]
        )
        by_id = {doc.case_id: doc for doc in corpus}

        candidates: list[DuplicateCandidate] = []
        for case_id, combined_score in combined:
            doc = by_id.get(case_id)
            if doc is None:
                continue
            evidence_output, evidence_event = self._evidence_tool.run(
                RetrieveCaseEvidenceInput(query_fields=query_fields, candidate=doc), ctx
            )
            events.append(evidence_event)
            candidates.append(
                DuplicateCandidate(
                    candidate_case_id=case_id,
                    bm25_score=bm25_scores.get(case_id, 0.0),
                    vector_score=vector_scores.get(case_id, 0.0),
                    combined_score=combined_score,
                    matching_fields=evidence_output.matching_fields,
                    conflicting_fields=evidence_output.conflicting_fields,
                    evidence_snippets=evidence_output.evidence_snippets,
                )
            )

        events.append(
            RetrievalEvent(
                case_id=ctx.case_id,
                trace_id=ctx.trace_id,
                query_summary=query_text[:200],
                candidates_returned=len(candidates),
                top_k=ranker.top_k,
            )
        )
        logger.debug(
            "duplicate_search case_id=%s provider=%s corpus_size=%d candidates=%d",
            ctx.case_id,
            self._embedding_provider.name,
            len(corpus),
            len(candidates),
        )
        return candidates, events
