"""Unit tests for src/memory/semantic.py."""

from __future__ import annotations

from src.evaluation.schemas import GoldenCase
from src.memory.semantic import (
    read_event_vocabulary,
    read_product_vocabulary,
    read_reference_case_corpus,
)
from src.models.enums import AgentName, MemoryOperation, MemoryTier
from src.tools.base import ToolContext

CTX = ToolContext(
    case_id="case_a",
    trace_id="trace_001",
    agent=AgentName.DUPLICATE_SEARCH,
    authorized_tools=frozenset(),
)


def test_read_product_vocabulary_returns_canonical_products_and_aliases() -> None:
    data, event = read_product_vocabulary(CTX)
    assert "DemoInsulex" in data["canonical_products"]
    assert data["aliases"]["Demo-Insulex"] == "DemoInsulex"
    assert event.tier == MemoryTier.SEMANTIC
    assert event.operation == MemoryOperation.READ


def test_read_event_vocabulary_returns_synonym_map() -> None:
    data, _ = read_event_vocabulary(CTX)
    assert data["throwing up repeatedly"] == "vomiting"


def test_read_reference_case_corpus_returns_golden_cases() -> None:
    cases, event = read_reference_case_corpus(CTX)
    assert len(cases) > 0
    assert all(isinstance(c, GoldenCase) for c in cases)
    assert event.key == "reference_case_corpus"
