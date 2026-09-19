"""Deterministic tool layer.

Every tool is a `BaseTool` subclass (see `src.tools.base`) grouped by
purpose into `intake`, `document`, `domain`, `retrieval`, `quality`, and
`workflow`. `src.tools.registry.ALL_TOOLS` is the full catalog; import
individual tool classes from their submodule (e.g.
``from src.tools.intake import ParseEmailTool``).
"""

from src.tools.base import (
    BaseTool,
    ToolAuthorizationError,
    ToolContext,
    ToolExecutionError,
    limit_results,
    run_tool,
)

__all__ = [
    "BaseTool",
    "ToolAuthorizationError",
    "ToolContext",
    "ToolExecutionError",
    "limit_results",
    "run_tool",
]
