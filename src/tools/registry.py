"""Tool registry: the source of truth for which tools exist and which
agent may call which tool.

Reads `config/config.yaml`'s `agents.<agent>.tool_allowlist` — the policy
lives in one place (config), not scattered across agent code — and turns
it into the `frozenset[str]` a `ToolContext` needs. Also exposes
`ALL_TOOLS`, the full registered-tool catalog, so guardrail code can
verify an agent's allowlist only references real tools.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.models.enums import AgentName
from src.tools.base import BaseTool
from src.tools.document import (
    AssessOcrQualityTool,
    CreatePageCitationsTool,
    ExtractPdfTextTool,
    NormalizeSourcePassagesTool,
    PerformOcrTool,
)
from src.tools.domain import (
    CheckMinimumCaseCriteriaTool,
    DetectExplicitTriageIndicatorsTool,
    LookupEventTermTool,
    LookupProductAliasTool,
    NormalizeDateTool,
    SuggestTriagePriorityTool,
    ValidateMedicalFieldTool,
)
from src.tools.intake import (
    EnforceFileSizeTool,
    EnforcePageLimitTool,
    ExtractAttachmentsTool,
    ParseEmailTool,
    SanitizeFilenameTool,
    ValidateFileSignatureTool,
    ValidateFileTool,
)
from src.tools.quality import (
    CompareReportWithFieldsTool,
    DetectConflictingValuesTool,
    DetectUnsupportedClaimsTool,
    ValidateCitationsTool,
    ValidateGoalCompletionTool,
)
from src.tools.retrieval import (
    Bm25SearchTool,
    ExactCaseSearchTool,
    MetadataFilterTool,
    ReciprocalRankFusionTool,
    RetrieveCaseEvidenceTool,
    VectorSearchTool,
)
from src.tools.workflow import (
    CreateReviewTaskTool,
    InMemoryCaseStateStore,
    PauseForHumanReviewTool,
    ReadCaseStateTool,
    RecordAuditEventTool,
    RouteCaseTool,
    WriteCaseCheckpointTool,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

# Shared so a write_case_checkpoint call is visible to a later read_case_state
# call within the same process -- each tool defaults to its own private store
# otherwise, which would make the Supervisor's checkpoint round-trip inert.
_CASE_STATE_STORE = InMemoryCaseStateStore()

_ALL_TOOL_INSTANCES: list[BaseTool[Any, Any]] = [
    # Intake
    ParseEmailTool(),
    ExtractAttachmentsTool(),
    ValidateFileTool(),
    ValidateFileSignatureTool(),
    SanitizeFilenameTool(),
    EnforceFileSizeTool(),
    EnforcePageLimitTool(),
    # Document
    ExtractPdfTextTool(),
    PerformOcrTool(),
    AssessOcrQualityTool(),
    CreatePageCitationsTool(),
    NormalizeSourcePassagesTool(),
    # Domain
    LookupProductAliasTool(),
    LookupEventTermTool(),
    NormalizeDateTool(),
    ValidateMedicalFieldTool(),
    CheckMinimumCaseCriteriaTool(),
    DetectExplicitTriageIndicatorsTool(),
    SuggestTriagePriorityTool(),
    # Retrieval
    ExactCaseSearchTool(),
    Bm25SearchTool(),
    VectorSearchTool(),
    MetadataFilterTool(),
    ReciprocalRankFusionTool(),
    RetrieveCaseEvidenceTool(),
    # Quality
    ValidateCitationsTool(),
    DetectUnsupportedClaimsTool(),
    CompareReportWithFieldsTool(),
    DetectConflictingValuesTool(),
    ValidateGoalCompletionTool(),
    # Workflow
    ReadCaseStateTool(store=_CASE_STATE_STORE),
    WriteCaseCheckpointTool(store=_CASE_STATE_STORE),
    PauseForHumanReviewTool(),
    RouteCaseTool(),
    RecordAuditEventTool(),
    CreateReviewTaskTool(),
]

ALL_TOOLS: dict[str, BaseTool[Any, Any]] = {tool.name: tool for tool in _ALL_TOOL_INSTANCES}


@lru_cache(maxsize=1)
def _load_agent_tool_allowlists() -> dict[str, list[str]]:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {
        agent_name: cfg.get("tool_allowlist", [])
        for agent_name, cfg in config.get("agents", {}).items()
    }


def authorized_tools_for(agent: AgentName) -> frozenset[str]:
    """The set of tool names `agent` may call, per config/config.yaml."""
    allowlists = _load_agent_tool_allowlists()
    return frozenset(allowlists.get(agent.value, []))


def validate_allowlists_reference_real_tools() -> list[str]:
    """Return a list of problems (empty = clean): any configured
    tool_allowlist entry that does not name a tool actually registered in
    ALL_TOOLS."""
    problems = []
    for agent_name, tools in _load_agent_tool_allowlists().items():
        for tool_name in tools:
            if tool_name not in ALL_TOOLS:
                problems.append(f"agent {agent_name!r} allowlists unknown tool {tool_name!r}")
    return problems
