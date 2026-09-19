"""Unit tests for config/settings.py."""

from __future__ import annotations

import os

from config.settings import DBBackend, LLMProvider, Settings, get_settings


def test_defaults_are_mock_llm_and_sqlite(monkeypatch) -> None:
    # Isolate from any real .env / environment the host machine may have set.
    for key in list(os.environ):
        if key.startswith(("LLM_", "DB_", "APP_", "ANTHROPIC_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("APP_ENV", "test")

    settings = get_settings()

    assert isinstance(settings, Settings)
    assert settings.llm_provider == LLMProvider.MOCK
    assert settings.db_backend == DBBackend.SQLITE
    assert settings.max_agent_retries == 1
    assert settings.secret_key  # non-empty default, must be overridden in real deploys


def test_env_var_overrides_default(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-opus-5")

    settings = get_settings()

    assert settings.llm_provider == LLMProvider.ANTHROPIC
    assert settings.llm_model == "claude-opus-5"
