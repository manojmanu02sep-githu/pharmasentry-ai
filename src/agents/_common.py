"""Shared helpers used by every agent module, so each agent implements only
its own bounded decision rather than repeating ToolContext/AgentEvent/
tool-call/citation-lookup boilerplate twelve times."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel

from config.settings import get_settings
from src.llm.base import LLMCallMetadata
from src.llm.factory import get_llm_provider
from src.models.audit import AgentEvent, ToolEvent
from src.models.enums import AgentName, ConflictStatus
from src.models.evidence import Citation, FieldValue, SourcePassage
from src.models.goal import SuccessCriterion
from src.models.reasoning import AgentDecision
from src.tools.base import ToolContext
from src.tools.registry import ALL_TOOLS, authorized_tools_for

ModelT = TypeVar("ModelT", bound=BaseModel)


def make_tool_context(case_id: str, trace_id: str, agent: AgentName) -> ToolContext:
    return ToolContext(
        case_id=case_id,
        trace_id=trace_id,
        agent=agent,
        authorized_tools=authorized_tools_for(agent),
    )


def call_tool(
    name: str, tool_input: BaseModel, ctx: ToolContext, tool_events: list[ToolEvent]
) -> Any:
    """Run an allowlisted tool by name via the shared registry and append
    its ToolEvent to the caller's event list. Raises ToolAuthorizationError
    if `name` isn't on `ctx.authorized_tools` -- callers never bypass the
    allowlist by constructing a tool directly."""
    output, event = ALL_TOOLS[name].run(tool_input, ctx)
    tool_events.append(event)
    return output


def make_agent_event(
    decision: AgentDecision,
    trace_id: str,
    node: str,
    model_version: str,
    prompt_version: str | None = None,
    latency_ms: float | None = None,
    retry_count: int = 0,
) -> AgentEvent:
    """Build the AgentEvent for one node's bounded decision. Only the
    concise `decision_summary` is persisted -- never chain-of-thought."""
    return AgentEvent(
        case_id=decision.case_id,
        trace_id=trace_id,
        agent=decision.agent,
        node=node,
        model_version=model_version,
        prompt_version=prompt_version,
        decision_summary=decision.decision_summary,
        latency_ms=latency_ms,
        retry_count=retry_count,
        timestamp=datetime.now(UTC),
    )


def first_passage_containing(
    passages: list[SourcePassage], substring: str
) -> SourcePassage | None:
    for passage in passages:
        if substring and substring in passage.text:
            return passage
    return None


def find_citations(passages: list[SourcePassage], quoted_text: str) -> list[Citation]:
    """Every passage that actually contains `quoted_text` verbatim, as a
    Citation. Returns [] when nothing grounds the text -- callers must
    never fabricate a citation for an ungrounded value."""
    if not quoted_text:
        return []
    return [
        Citation(passage_id=p.passage_id, quoted_text=quoted_text, page_number=p.page_number)
        for p in passages
        if quoted_text in p.text
    ]


def call_llm_or_none(
    response_model: type[ModelT],
    context: dict[str, Any],
    prompt: str = "",
) -> tuple[ModelT | None, LLMCallMetadata | None]:
    """Call the configured LLM provider; return (None, None) instead of
    raising when it's unavailable, so callers can fall back to a
    deterministic route per CLAUDE.md's "LLM unavailable" conditional
    route rather than crashing the node."""
    try:
        provider = get_llm_provider(get_settings())
        return provider.complete(prompt, response_model, context=context)
    except Exception:
        return None, None


def new_success_criterion(criterion_id: str, description: str, met: bool) -> SuccessCriterion:
    return SuccessCriterion(criterion_id=criterion_id, description=description, met=met)


def make_field_value(
    field_name: str,
    value: str | None,
    passages: list[SourcePassage],
    confidence: float,
    conflicting_values: list[str] | None = None,
    citation_text: str | None = None,
) -> FieldValue:
    """Build a FieldValue grounded in `passages`. A value with no matching
    passage is kept (never dropped) but its confidence is capped low, so a
    reviewer can see it was asserted without direct textual support.

    `citation_text` lets a caller ground the citation search in the exact
    substring that appeared in the source (e.g. the raw alias "DemoGluca"
    or the matched phrase "hospitalized") when `value` itself is a
    normalized/canonical or boolean-token form that never appears
    verbatim in the text -- without this, a correctly normalized value
    would be wrongly flagged as an unsupported claim.
    """
    if value is None:
        return FieldValue(
            field_name=field_name,
            value=None,
            citations=[],
            confidence=0.0,
            conflict_status=ConflictStatus.MISSING,
        )
    citations = find_citations(passages, citation_text if citation_text is not None else value)
    grounded_confidence = confidence if citations else min(confidence, 0.2)
    conflict_status = ConflictStatus.CONFLICTING if conflicting_values else ConflictStatus.NONE
    return FieldValue(
        field_name=field_name,
        value=value,
        citations=citations,
        confidence=round(grounded_confidence, 2),
        conflict_status=conflict_status,
        conflicting_values=conflicting_values or [],
    )
