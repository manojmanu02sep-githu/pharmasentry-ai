"""Unit tests for src/tools/registry.py."""

from __future__ import annotations

from src.models.enums import AgentName
from src.tools.registry import (
    ALL_TOOLS,
    authorized_tools_for,
    validate_allowlists_reference_real_tools,
)

EXPECTED_TOOL_NAMES = {
    # Intake
    "parse_email", "extract_attachments", "validate_file", "validate_file_signature",
    "sanitize_filename", "enforce_file_size", "enforce_page_limit",
    # Document
    "extract_pdf_text", "perform_ocr", "assess_ocr_quality", "create_page_citations",
    "normalize_source_passages",
    # Domain
    "lookup_product_alias", "lookup_event_term", "normalize_date", "validate_medical_field",
    "check_minimum_case_criteria", "detect_explicit_triage_indicators", "suggest_triage_priority",
    # Retrieval
    "exact_case_search", "bm25_search", "vector_search", "metadata_filter",
    "reciprocal_rank_fusion", "retrieve_case_evidence",
    # Quality
    "validate_citations", "detect_unsupported_claims", "compare_report_with_fields",
    "detect_conflicting_values", "validate_goal_completion",
    # Workflow
    "read_case_state", "write_case_checkpoint", "pause_for_human_review", "route_case",
    "record_audit_event", "create_review_task",
}


def test_all_36_required_tools_are_registered() -> None:
    assert EXPECTED_TOOL_NAMES <= ALL_TOOLS.keys()
    assert len(ALL_TOOLS) == 36


def test_every_agent_allowlist_references_only_real_tools() -> None:
    assert validate_allowlists_reference_real_tools() == []


def test_authorized_tools_for_duplicate_search_excludes_reporter_context_tools() -> None:
    """Context Isolation and Delegation: the Duplicate Agent's tool
    allowlist must not include anything that would let it read reporter
    contact details."""
    tools = authorized_tools_for(AgentName.DUPLICATE_SEARCH)
    assert "bm25_search" in tools
    assert "vector_search" in tools
    assert "reciprocal_rank_fusion" in tools
    assert "read_case_state" not in tools  # would expose full case state, incl. reporter info


def test_every_agent_has_a_non_empty_allowlist() -> None:
    for agent in AgentName:
        assert authorized_tools_for(agent), f"{agent.value} has no configured tool allowlist"
