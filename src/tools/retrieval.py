"""Retrieval tools: exact/lexical/vector search, metadata filtering, and
reciprocal rank fusion.

These implement the TOOL INTERFACE and a real, deterministic, offline
working implementation. The persistent FAISS/BM25 index
(`src/retrieval/faiss_index.py`, `bm25_index.py`, built via
`scripts/build_retrieval_index.py`) and the pluggable embedding provider
(`src/retrieval/embeddings.py`) build on this same interface —
`VectorSearchTool` accepts an optional `embed_fn` so callers can swap in
sentence-transformers embeddings without changing this tool's shape.

Corpus documents are passed in explicitly (`CorpusDocument`) rather than
read from a hidden global index, keeping every tool here a pure function
of its input — easy to unit test, no hidden state.
"""

from __future__ import annotations

import math
import re
import zlib
from collections.abc import Callable

from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from src.tools.base import BaseTool, ToolContext, limit_results

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def tokenize(text: str) -> list[str]:
    """Public wrapper around the module's tokenizer, for reuse by
    src/retrieval (Phase 5) without reaching into a private name."""
    return _tokenize(text)


class CorpusDocument(BaseModel):
    case_id: str
    text: str = ""
    fields: dict[str, str | None] = Field(default_factory=dict)


# --- exact_case_search --------------------------------------------------------


class ExactCaseSearchInput(BaseModel):
    query_fields: dict[str, str | None]
    corpus: list[CorpusDocument]


class ExactCaseSearchOutput(BaseModel):
    matching_case_ids: list[str] = Field(default_factory=list)


class ExactCaseSearchTool(BaseTool[ExactCaseSearchInput, ExactCaseSearchOutput]):
    name = "exact_case_search"
    purpose = "Find corpus cases whose fields exactly match every non-null query field."
    timeout_seconds = 3.0
    max_retries = 2

    def _execute(
        self, tool_input: ExactCaseSearchInput, ctx: ToolContext
    ) -> ExactCaseSearchOutput:
        query = {k: v for k, v in tool_input.query_fields.items() if v is not None}
        matches: list[str] = []
        for doc in tool_input.corpus:
            if _fields_match(doc.fields, query):
                matches.append(doc.case_id)
        limited, _ = limit_results(matches, 50)
        return ExactCaseSearchOutput(matching_case_ids=limited)


def _fields_match(doc_fields: dict[str, str | None], query: dict[str, str]) -> bool:
    for key, value in query.items():
        doc_value = doc_fields.get(key)
        if doc_value is None or doc_value.strip().lower() != value.strip().lower():
            return False
    return True


# --- bm25_search --------------------------------------------------------


class Bm25SearchInput(BaseModel):
    query_text: str = Field(max_length=100_000)
    corpus: list[CorpusDocument]
    top_k: int = 5


class ScoredCandidate(BaseModel):
    case_id: str
    score: float


class Bm25SearchOutput(BaseModel):
    results: list[ScoredCandidate] = Field(default_factory=list)


class Bm25SearchTool(BaseTool[Bm25SearchInput, Bm25SearchOutput]):
    name = "bm25_search"
    purpose = "Lexical (BM25) ranking of the corpus against a query text."
    timeout_seconds = 5.0
    max_retries = 1

    def _execute(self, tool_input: Bm25SearchInput, ctx: ToolContext) -> Bm25SearchOutput:
        if not tool_input.corpus:
            return Bm25SearchOutput(results=[])
        tokenized_corpus = [_tokenize(doc.text) for doc in tool_input.corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(_tokenize(tool_input.query_text))
        ranked = sorted(
            zip(tool_input.corpus, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        results = [
            ScoredCandidate(case_id=doc.case_id, score=round(float(score), 6))
            for doc, score in ranked[: tool_input.top_k]
        ]
        return Bm25SearchOutput(results=results)


# --- vector_search --------------------------------------------------------

_EMBEDDING_DIM = 128


def deterministic_embedding(text: str, dim: int = _EMBEDDING_DIM) -> list[float]:
    """A deterministic, offline "test embedding provider": hashed
    bag-of-words, L2-normalized. Not semantically meaningful the way a
    real sentence-transformers model is — it exists so vector_search has a
    real, reproducible implementation with no model download and no
    network access. Phase 5 swaps in sentence-transformers behind the
    same function signature for production-quality similarity.
    """
    vector = [0.0] * dim
    for token in _tokenize(text):
        idx = zlib.crc32(token.encode("utf-8")) % dim
        vector[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]
    return vector


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class VectorSearchInput(BaseModel):
    query_text: str = Field(max_length=100_000)
    corpus: list[CorpusDocument]
    top_k: int = 5


class VectorSearchOutput(BaseModel):
    results: list[ScoredCandidate] = Field(default_factory=list)


class VectorSearchTool(BaseTool[VectorSearchInput, VectorSearchOutput]):
    name = "vector_search"
    purpose = "Embedding-similarity ranking of the corpus against a query text."
    timeout_seconds = 5.0
    max_retries = 1

    def __init__(self, embed_fn: Callable[[str], list[float]] | None = None) -> None:
        super().__init__()
        self._embed = embed_fn or deterministic_embedding

    def _execute(self, tool_input: VectorSearchInput, ctx: ToolContext) -> VectorSearchOutput:
        if not tool_input.corpus:
            return VectorSearchOutput(results=[])
        query_vec = self._embed(tool_input.query_text)
        scored = [
            (doc.case_id, _cosine(query_vec, self._embed(doc.text))) for doc in tool_input.corpus
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        results = [
            ScoredCandidate(case_id=cid, score=round(score, 6))
            for cid, score in scored[: tool_input.top_k]
        ]
        return VectorSearchOutput(results=results)


# --- metadata_filter --------------------------------------------------------


class MetadataFilterInput(BaseModel):
    candidate_case_ids: list[str]
    corpus: list[CorpusDocument]
    filters: dict[str, str]  # field -> required value


class MetadataFilterOutput(BaseModel):
    filtered_case_ids: list[str] = Field(default_factory=list)


class MetadataFilterTool(BaseTool[MetadataFilterInput, MetadataFilterOutput]):
    name = "metadata_filter"
    purpose = "Narrow a candidate list to those matching required metadata field values."
    timeout_seconds = 2.0
    max_retries = 2

    def _execute(
        self, tool_input: MetadataFilterInput, ctx: ToolContext
    ) -> MetadataFilterOutput:
        by_id = {doc.case_id: doc for doc in tool_input.corpus}
        kept = [
            cid
            for cid in tool_input.candidate_case_ids
            if cid in by_id
            and all(by_id[cid].fields.get(k) == v for k, v in tool_input.filters.items())
        ]
        return MetadataFilterOutput(filtered_case_ids=kept)


# --- reciprocal_rank_fusion --------------------------------------------------------


class ReciprocalRankFusionInput(BaseModel):
    rankings: list[list[str]]  # one ranked case_id list per retriever
    k: int = 60
    top_k: int = 5


class ReciprocalRankFusionOutput(BaseModel):
    results: list[ScoredCandidate] = Field(default_factory=list)


class ReciprocalRankFusionTool(
    BaseTool[ReciprocalRankFusionInput, ReciprocalRankFusionOutput]
):
    name = "reciprocal_rank_fusion"
    purpose = (
        "Fuse multiple ranked candidate lists (e.g. BM25 + vector) into one "
        "ranking via the standard Reciprocal Rank Fusion formula: "
        "score(d) = sum(1 / (k + rank))."
    )
    timeout_seconds = 2.0
    max_retries = 2

    def _execute(
        self, tool_input: ReciprocalRankFusionInput, ctx: ToolContext
    ) -> ReciprocalRankFusionOutput:
        scores: dict[str, float] = {}
        for ranking in tool_input.rankings:
            for rank, case_id in enumerate(ranking):
                scores[case_id] = scores.get(case_id, 0.0) + 1.0 / (tool_input.k + rank + 1)
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        results = [
            ScoredCandidate(case_id=cid, score=round(score, 6))
            for cid, score in ranked[: tool_input.top_k]
        ]
        return ReciprocalRankFusionOutput(results=results)


# --- retrieve_case_evidence --------------------------------------------------------


class RetrieveCaseEvidenceInput(BaseModel):
    query_fields: dict[str, str | None]
    candidate: CorpusDocument


class RetrieveCaseEvidenceOutput(BaseModel):
    matching_fields: list[str] = Field(default_factory=list)
    conflicting_fields: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)


class RetrieveCaseEvidenceTool(
    BaseTool[RetrieveCaseEvidenceInput, RetrieveCaseEvidenceOutput]
):
    name = "retrieve_case_evidence"
    purpose = (
        "Explain WHY a candidate matched: which fields agree, which "
        "conflict, and a short evidence snippet — the Duplicate Agent "
        "never merges, it only ever explains and defers to human review."
    )
    timeout_seconds = 2.0
    max_retries = 2

    def _execute(
        self, tool_input: RetrieveCaseEvidenceInput, ctx: ToolContext
    ) -> RetrieveCaseEvidenceOutput:
        matching, conflicting = [], []
        for field, query_value in tool_input.query_fields.items():
            if query_value is None:
                continue
            candidate_value = tool_input.candidate.fields.get(field)
            if candidate_value is None:
                continue
            if candidate_value.strip().lower() == query_value.strip().lower():
                matching.append(field)
            else:
                conflicting.append(field)

        snippet = tool_input.candidate.text[:300]
        return RetrieveCaseEvidenceOutput(
            matching_fields=matching,
            conflicting_fields=conflicting,
            evidence_snippets=[snippet] if snippet else [],
        )
