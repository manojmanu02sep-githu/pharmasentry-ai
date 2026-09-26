"""Deterministic mock LLM provider — no network calls, no randomness.

Fills each known response model from the structured ``context`` dict
(never by parsing the ``prompt`` string), so output is fully reproducible
across runs and safe for tests and CI.
"""

from __future__ import annotations

import time
from typing import Any, TypeVar

from pydantic import BaseModel

from src.llm.base import LLMCallMetadata, LLMUnavailableError
from src.llm.schemas import NarrativeDraftResult, QualityReviewResult, SafetyClassificationResult

T = TypeVar("T", bound=BaseModel)

_NON_SAFETY_KEYWORDS = (
    "unsubscribe",
    "newsletter",
    "invoice",
    "marketing",
    "promotional offer",
    "job opening",
)

_SAFETY_KEYWORDS = (
    "adverse event",
    "adverse reaction",
    "side effect",
    "hospitalized",
    "hospitalised",
    "reaction",
    "overdose",
    "pancreatitis",
    "safety report",
)


def _classify_safety(context: dict[str, Any]) -> SafetyClassificationResult:
    text = f"{context.get('email_subject', '')} {context.get('email_body', '')}".lower()
    if any(kw in text for kw in _NON_SAFETY_KEYWORDS):
        return SafetyClassificationResult(
            is_safety_report=False,
            confidence=0.9,
            rationale="Matched non-safety keyword(s) in subject/body.",
        )
    matched = [kw for kw in _SAFETY_KEYWORDS if kw in text]
    if matched:
        return SafetyClassificationResult(
            is_safety_report=True,
            confidence=min(0.6 + 0.1 * len(matched), 0.95),
            rationale=f"Matched safety keyword(s): {', '.join(matched)}.",
        )
    return SafetyClassificationResult(
        is_safety_report=False,
        confidence=0.5,
        rationale="No explicit safety or non-safety keywords matched.",
    )


def _draft_narrative(context: dict[str, Any]) -> NarrativeDraftResult:
    facts: list[str] = context.get("facts", [])
    if not facts:
        return NarrativeDraftResult(narrative_text="")
    return NarrativeDraftResult(narrative_text=" ".join(facts))


def _review_quality(context: dict[str, Any]) -> QualityReviewResult:
    checks: list[dict[str, Any]] = context.get("checks", [])
    failed = [c for c in checks if not c.get("passed", True)]
    if not failed:
        return QualityReviewResult(passed=True, feedback="All quality checks passed.")
    detail = "; ".join(f"{c.get('name', 'check')}: {c.get('detail', 'failed')}" for c in failed)
    return QualityReviewResult(passed=False, feedback=f"Failed checks: {detail}")


class MockLLMProvider:
    """Deterministic, rule-based stand-in for a real LLM. Used whenever
    ``settings.llm_provider == LLMProvider.MOCK`` (the default), including
    in every test."""

    name = "mock"
    model = "mock-deterministic-v1"

    def complete(
        self,
        prompt: str,
        response_model: type[T],
        *,
        context: dict[str, Any] | None = None,
    ) -> tuple[T, LLMCallMetadata]:
        del prompt  # never parsed: dispatch is on structured context only
        ctx = context or {}
        started = time.monotonic()

        result: BaseModel
        if response_model is SafetyClassificationResult:
            result = _classify_safety(ctx)
        elif response_model is NarrativeDraftResult:
            result = _draft_narrative(ctx)
        elif response_model is QualityReviewResult:
            result = _review_quality(ctx)
        else:
            raise LLMUnavailableError(
                f"MockLLMProvider has no deterministic rule for {response_model!r}"
            )

        latency_ms = (time.monotonic() - started) * 1000
        metadata = LLMCallMetadata(
            provider_name=self.name,
            model=self.model,
            tokens_used=0,
            cost_usd=0.0,
            latency_ms=latency_ms,
        )
        return result, metadata  # type: ignore[return-value]
