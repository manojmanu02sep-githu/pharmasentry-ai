"""Observability/traceability event models — the complete audit trace.

Every event carries case_id and trace_id. Only concise decision summaries
are stored; raw chain-of-thought is never persisted (Absolute Boundaries).
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from src.models.enums import AgentName, MemoryOperation, MemoryTier, ToolCallStatus


class TraceContext(BaseModel):
    case_id: str
    trace_id: str


class AgentEvent(BaseModel):
    case_id: str
    trace_id: str
    agent: AgentName
    node: str
    model_version: str | None = None
    prompt_version: str | None = None
    decision_summary: str
    latency_ms: float | None = None
    tokens_used: int | None = None
    cost_usd: float | None = None
    retry_count: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolEvent(BaseModel):
    case_id: str
    trace_id: str
    tool_name: str
    agent: AgentName
    status: ToolCallStatus
    input_summary: str
    output_summary: str
    latency_ms: float | None = None
    authorized: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryEvent(BaseModel):
    case_id: str
    trace_id: str
    tier: MemoryTier
    operation: MemoryOperation
    key: str
    agent: AgentName
    authorized: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RetrievalEvent(BaseModel):
    case_id: str
    trace_id: str
    query_summary: str
    candidates_returned: int
    top_k: int
    latency_ms: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ErrorRecord(BaseModel):
    case_id: str
    trace_id: str
    node: str
    error_type: str
    message: str
    recoverable: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
