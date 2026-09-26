"""Unit tests for src/memory/procedural.py."""

from __future__ import annotations

from src.memory.procedural import (
    read_limits,
    read_model_and_prompt_versions,
    read_node_order,
    read_retention_policy,
    read_retrieval_policy,
    read_tool_allowlist,
)
from src.models.enums import AgentName, MemoryOperation, MemoryTier
from src.tools.base import ToolContext

CTX = ToolContext(
    case_id="case_a",
    trace_id="trace_001",
    agent=AgentName.SUPERVISOR,
    authorized_tools=frozenset(),
)


def test_read_node_order_matches_config() -> None:
    node_order, event = read_node_order(CTX)
    assert node_order[0] == "goal_agent"
    assert node_order[-1] == "human_review"
    assert event.tier == MemoryTier.PROCEDURAL
    assert event.operation == MemoryOperation.READ


def test_read_tool_allowlist_for_duplicate_search_excludes_read_case_state() -> None:
    allowlist, _ = read_tool_allowlist(AgentName.DUPLICATE_SEARCH, CTX)
    assert "bm25_search" in allowlist
    assert "vector_search" in allowlist
    assert "read_case_state" not in allowlist  # reporter contact isolation


def test_read_limits_matches_config() -> None:
    limits, _ = read_limits(CTX)
    assert limits["max_delegation_depth"] == 6
    assert limits["max_agent_retries"] == 1


def test_read_retrieval_policy_matches_config() -> None:
    policy, _ = read_retrieval_policy(CTX)
    assert policy["bm25_weight"] == 0.5
    assert policy["vector_weight"] == 0.5
    assert policy["top_k"] == 5


def test_read_retention_policy_matches_config() -> None:
    retention, _ = read_retention_policy(CTX)
    assert retention["default_retention_days"] == 90


def test_read_model_and_prompt_versions_matches_config() -> None:
    versions, _ = read_model_and_prompt_versions(CTX)
    assert versions["model_version"] == "phase1-scaffold"
    assert versions["prompt_version"] == "v1"
