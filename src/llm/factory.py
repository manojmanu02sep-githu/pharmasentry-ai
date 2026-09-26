"""Selects the configured LLM provider.

Only the mock provider is implemented in this build. Selecting the
Anthropic provider raises ``LLMUnavailableError`` rather than fabricating
a client, so callers route to the deterministic fallback or manual review
(CLAUDE.md's "LLM unavailable" conditional route).
"""

from __future__ import annotations

from config.settings import LLMProvider, Settings
from src.llm.base import LLMProviderProtocol, LLMUnavailableError
from src.llm.mock import MockLLMProvider


def get_llm_provider(settings: Settings) -> LLMProviderProtocol:
    if settings.llm_provider == LLMProvider.MOCK:
        return MockLLMProvider()
    raise LLMUnavailableError(
        f"LLM provider {settings.llm_provider.value!r} is not implemented in this "
        "build; set LLM_PROVIDER=mock or route this case to manual review."
    )
