"""Typed application settings loaded from environment variables / .env.

Deterministic, code-level configuration (as opposed to LLM-driven decisions).
See .env.example for every supported variable and its default.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class LLMProvider(str, Enum):
    MOCK = "mock"
    ANTHROPIC = "anthropic"


class DBBackend(str, Enum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"


class OCRProvider(str, Enum):
    LOCAL = "local"


class EmbeddingProviderName(str, Enum):
    DETERMINISTIC = "deterministic"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_env: str = "local"
    log_level: str = "INFO"
    secret_key: str = "change-me-to-a-random-value"

    # LLM provider
    llm_provider: LLMProvider = LLMProvider.MOCK
    anthropic_api_key: str = ""
    llm_model: str = "claude-opus-5"
    llm_max_retries: int = 2
    llm_timeout_seconds: int = 30

    # Database
    db_backend: DBBackend = DBBackend.SQLITE
    sqlite_path: str = "./data/pharmasentry.db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "pharmasentry"
    postgres_user: str = "pharmasentry"
    postgres_password: str = ""

    # Retrieval
    vector_index_path: str = "./data/vector_index"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_provider: EmbeddingProviderName = EmbeddingProviderName.DETERMINISTIC
    bm25_index_path: str = "./data/bm25_index"

    # OCR
    ocr_provider: OCRProvider = OCRProvider.LOCAL

    # Observability
    otel_exporter_otlp_endpoint: str = ""
    prometheus_port: int = 9090
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""

    # Governance / limits
    max_delegation_depth: int = Field(default=6, ge=1)
    max_agent_retries: int = Field(default=1, ge=0)
    max_runtime_seconds: int = Field(default=300, ge=1)
    max_tokens_per_case: int = Field(default=200_000, ge=1)
    retention_days: int = Field(default=90, ge=1)


def get_settings() -> Settings:
    """Return a freshly loaded Settings instance.

    Not cached: tests routinely monkeypatch environment variables between
    cases and must see the updated values.
    """
    return Settings()
