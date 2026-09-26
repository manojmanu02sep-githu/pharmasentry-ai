"""The four memory tiers required by CLAUDE.md's Memory section.

- Working: `src.memory.working` — logged access to the in-flight CaseState.
- Episodic: `src.memory.episodic` — durable, case-isolated SQLite history.
- Semantic: `src.memory.semantic` — read-only synthetic reference content.
- Procedural: `src.memory.procedural` — read-only workflow policy/config.

Every accessor returns (result, MemoryEvent) so callers append the event
to `CaseState.memory_events` — no accessor here persists chain-of-thought
or secrets, only concise keys and structured results.
"""

from __future__ import annotations

from src.memory.episodic import (
    EpisodicHistoryEntry,
    EpisodicMemoryStore,
    read_episodic_history,
    read_episodic_state,
    record_episodic_event,
    write_episodic_state,
)
from src.memory.procedural import (
    read_limits,
    read_model_and_prompt_versions,
    read_node_order,
    read_retention_policy,
    read_retrieval_policy,
    read_tool_allowlist,
)
from src.memory.semantic import (
    read_event_vocabulary,
    read_product_vocabulary,
    read_reference_case_corpus,
)
from src.memory.working import read_working_memory, write_working_memory

__all__ = [
    "EpisodicHistoryEntry",
    "EpisodicMemoryStore",
    "read_episodic_history",
    "read_episodic_state",
    "read_event_vocabulary",
    "read_limits",
    "read_model_and_prompt_versions",
    "read_node_order",
    "read_product_vocabulary",
    "read_reference_case_corpus",
    "read_retention_policy",
    "read_retrieval_policy",
    "read_tool_allowlist",
    "read_working_memory",
    "record_episodic_event",
    "write_episodic_state",
    "write_working_memory",
]
