"""Procedural memory: read-only access to prompts, schemas, policies, and
workflow instructions — i.e. `config/config.yaml`, the project's single
source of policy truth (see that file's own header comment: "This file is
procedural-memory-adjacent: it declares policy, not code").

Mirrors `src.tools.registry`'s config-loading pattern (`lru_cache` +
`yaml.safe_load`) rather than re-parsing ad hoc, so both modules stay in
sync with one on-disk file. Every accessor is logged via a MemoryEvent, no
case-isolation check applies (procedural memory is global policy, not
per-case data).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.models.audit import MemoryEvent
from src.models.enums import AgentName, MemoryOperation, MemoryTier
from src.tools.base import ToolContext

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _event(key: str, ctx: ToolContext) -> MemoryEvent:
    return MemoryEvent(
        case_id=ctx.case_id,
        trace_id=ctx.trace_id,
        tier=MemoryTier.PROCEDURAL,
        operation=MemoryOperation.READ,
        key=key,
        agent=ctx.agent,
    )


def read_node_order(ctx: ToolContext) -> tuple[list[str], MemoryEvent]:
    node_order = list(_load_config().get("graph", {}).get("node_order", []))
    return node_order, _event("graph.node_order", ctx)


def read_tool_allowlist(agent_name: AgentName, ctx: ToolContext) -> tuple[list[str], MemoryEvent]:
    agents_cfg = _load_config().get("agents", {})
    allowlist = list(agents_cfg.get(agent_name.value, {}).get("tool_allowlist", []))
    return allowlist, _event(f"agents.{agent_name.value}.tool_allowlist", ctx)


def read_limits(ctx: ToolContext) -> tuple[dict[str, int], MemoryEvent]:
    limits = dict(_load_config().get("limits", {}))
    return limits, _event("limits", ctx)


def read_retrieval_policy(ctx: ToolContext) -> tuple[dict[str, float | int], MemoryEvent]:
    policy = dict(_load_config().get("retrieval", {}))
    return policy, _event("retrieval", ctx)


def read_retention_policy(ctx: ToolContext) -> tuple[dict[str, int], MemoryEvent]:
    retention = dict(_load_config().get("retention", {}))
    return retention, _event("retention", ctx)


def read_model_and_prompt_versions(ctx: ToolContext) -> tuple[dict[str, str], MemoryEvent]:
    config = _load_config()
    versions = {
        "model_version": config.get("model_version", ""),
        "prompt_version": config.get("prompt_version", ""),
    }
    return versions, _event("model_and_prompt_versions", ctx)
