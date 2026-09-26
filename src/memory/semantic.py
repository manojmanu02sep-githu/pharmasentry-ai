"""Semantic memory: read-only access to synthetic reference content —
product aliases, event vocabulary, and the synthetic case corpus.

This is a thin, logged wrapper over data that already exists as a single
source of truth elsewhere (`src.tools.reference_data`, the golden dataset
via `src.evaluation.golden_loader`) — it does not duplicate or fork that
data, per the project's "extend, don't duplicate" rule and the standing
"do not modify the dataset" constraint (this module never writes to the
golden dataset; `load_all_cases()` is read-only and lru-cached upstream).

Semantic memory is global reference content, not case-specific state, so
there is no case-isolation check here — access is still logged via
MemoryEvent for observability, per CLAUDE.md's Memory section. Context
isolation for WHICH fields an agent may see (e.g. never reporter contact
details for the Duplicate Agent) is enforced downstream in
`src.retrieval.corpus`, not here.
"""

from __future__ import annotations

from src.evaluation.golden_loader import load_all_cases
from src.evaluation.schemas import GoldenCase
from src.models.audit import MemoryEvent
from src.models.enums import MemoryOperation, MemoryTier
from src.tools.base import ToolContext
from src.tools.reference_data import CANONICAL_PRODUCTS, EVENT_SYNONYMS, PRODUCT_ALIASES


def _event(key: str, ctx: ToolContext) -> MemoryEvent:
    return MemoryEvent(
        case_id=ctx.case_id,
        trace_id=ctx.trace_id,
        tier=MemoryTier.SEMANTIC,
        operation=MemoryOperation.READ,
        key=key,
        agent=ctx.agent,
    )


def read_product_vocabulary(
    ctx: ToolContext,
) -> tuple[dict[str, tuple[str, ...] | dict[str, str]], MemoryEvent]:
    data: dict[str, tuple[str, ...] | dict[str, str]] = {
        "canonical_products": CANONICAL_PRODUCTS,
        "aliases": PRODUCT_ALIASES,
    }
    return data, _event("product_vocabulary", ctx)


def read_event_vocabulary(ctx: ToolContext) -> tuple[dict[str, str], MemoryEvent]:
    return dict(EVENT_SYNONYMS), _event("event_vocabulary", ctx)


def read_reference_case_corpus(ctx: ToolContext) -> tuple[tuple[GoldenCase, ...], MemoryEvent]:
    """The full synthetic case corpus, for retrieval/duplicate search.

    Read-only: goes through `load_all_cases()` only, never regenerates or
    mutates the golden dataset.
    """
    cases = load_all_cases()
    return cases, _event("reference_case_corpus", ctx)
