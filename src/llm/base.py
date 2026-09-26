"""Provider-agnostic LLM interface: Pydantic structured output only.

Named ``LLMProviderProtocol`` (not ``LLMProvider``) to avoid colliding with
``config.settings.LLMProvider``, the enum selecting which implementation to
use.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMUnavailableError(RuntimeError):
    """Raised when no usable LLM provider can service a call.

    Callers must route to the deterministic fallback or manual review per
    CLAUDE.md's "LLM unavailable" conditional route — never fabricate a
    response.
    """


class LLMCallMetadata(BaseModel):
    provider_name: str
    model: str
    tokens_used: int | None = None
    cost_usd: float | None = None
    latency_ms: float


class LLMProviderProtocol(Protocol):
    def complete(
        self,
        prompt: str,
        response_model: type[T],
        *,
        context: dict[str, Any] | None = None,
    ) -> tuple[T, LLMCallMetadata]: ...
