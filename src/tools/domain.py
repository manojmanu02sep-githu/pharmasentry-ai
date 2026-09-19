"""Domain tools: product/event vocabulary lookups, date normalization,
field-level validation, minimum-criteria checking, and explicit
triage-indicator detection with a deterministic suggested-priority mapping.

Everything here is deterministic rule-/lookup-based code — no LLM calls —
per CLAUDE.md's "use deterministic code for ... exact rules, thresholds"
requirement.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from src.models.criteria import MinimumCriteriaResult
from src.models.enums import TriageIndicator, TriagePriority
from src.tools.base import BaseTool, ToolContext, limit_results
from src.tools.reference_data import EVENT_SYNONYMS, PRODUCT_ALIASES

# --- lookup_product_alias ----------------------------------------------------


class LookupProductAliasInput(BaseModel):
    text: str = Field(max_length=10_000)


class LookupProductAliasOutput(BaseModel):
    canonical_name: str | None = None
    alias_matched: str | None = None
    confidence: float = 0.0


class LookupProductAliasTool(BaseTool[LookupProductAliasInput, LookupProductAliasOutput]):
    name = "lookup_product_alias"
    purpose = "Resolve a fictional product alias/brand variant to its canonical name."
    timeout_seconds = 1.0
    max_retries = 2

    def _execute(
        self, tool_input: LookupProductAliasInput, ctx: ToolContext
    ) -> LookupProductAliasOutput:
        for alias, canonical in PRODUCT_ALIASES.items():
            if alias in tool_input.text:
                return LookupProductAliasOutput(
                    canonical_name=canonical, alias_matched=alias, confidence=1.0
                )
        return LookupProductAliasOutput()


# --- lookup_event_term --------------------------------------------------------


class LookupEventTermInput(BaseModel):
    text: str = Field(max_length=10_000)


class LookupEventTermOutput(BaseModel):
    canonical_term: str | None = None
    synonym_matched: str | None = None
    confidence: float = 0.0


class LookupEventTermTool(BaseTool[LookupEventTermInput, LookupEventTermOutput]):
    name = "lookup_event_term"
    purpose = "Resolve a colloquial event-description synonym to its canonical term."
    timeout_seconds = 1.0
    max_retries = 2

    def _execute(
        self, tool_input: LookupEventTermInput, ctx: ToolContext
    ) -> LookupEventTermOutput:
        for synonym, canonical in EVENT_SYNONYMS.items():
            if synonym in tool_input.text:
                return LookupEventTermOutput(
                    canonical_term=canonical, synonym_matched=synonym, confidence=1.0
                )
        return LookupEventTermOutput()


# --- normalize_date ------------------------------------------------------------

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}


def _build_ordinal_words() -> dict[str, int]:
    ones = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
            "eighth", "ninth"]
    teens = ["tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
             "sixteenth", "seventeenth", "eighteenth", "nineteenth"]
    tens_cardinal = {20: "twenty", 30: "thirty"}
    words: dict[str, int] = {}
    for i, w in enumerate(ones):
        if w:
            words[w] = i
    for i, w in enumerate(teens):
        words[w] = 10 + i
    words["twentieth"] = 20
    words["thirtieth"] = 30
    for base, prefix in tens_cardinal.items():
        for i in range(1, 10 if base == 20 else 2):
            words[f"{prefix}-{ones[i]}"] = base + i
    return words


_ORDINAL_WORDS = _build_ordinal_words()

# This offline generator/tool anchors any date with no explicit year to
# 2026 — the fixed synthetic reporting year used throughout this demo
# dataset (see scripts/generate_golden_dataset.py). A real deployment
# would infer the year from case-received context instead of a constant.
_DEFAULT_SYNTHETIC_YEAR = 2026

_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DOTTED_RE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})$")
_SLASH_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$")
_NON_STANDARD_ORDER_RE = re.compile(r"^\d{1,2}-\d{4}-\d{1,2}$")
_ORDINAL_OF_MONTH_RE = re.compile(r"the ([a-z-]+) of ([a-z]+)", re.IGNORECASE)


class NormalizeDateInput(BaseModel):
    raw_date_text: str = Field(max_length=200)


class NormalizeDateOutput(BaseModel):
    normalized_date: str | None = None
    status: Literal["ok", "ambiguous", "invalid"]
    notes: str = ""


class NormalizeDateTool(BaseTool[NormalizeDateInput, NormalizeDateOutput]):
    name = "normalize_date"
    purpose = (
        "Deterministically normalize a date string to ISO 8601, or flag it "
        "as ambiguous/invalid rather than silently guessing."
    )
    timeout_seconds = 1.0
    max_retries = 2

    def _execute(
        self, tool_input: NormalizeDateInput, ctx: ToolContext
    ) -> NormalizeDateOutput:
        text = tool_input.raw_date_text.strip()

        if m := _ISO_RE.match(text):
            year, month, day = (int(x) for x in m.groups())
            if _valid_ymd(year, month, day):
                return NormalizeDateOutput(normalized_date=text, status="ok")
            return NormalizeDateOutput(status="invalid", notes="ISO-shaped but out-of-range")

        if m := _DOTTED_RE.match(text):
            year, month, day = (int(x) for x in m.groups())
            if _valid_ymd(year, month, day):
                return NormalizeDateOutput(
                    normalized_date=f"{year:04d}-{month:02d}-{day:02d}", status="ok"
                )
            return NormalizeDateOutput(status="invalid", notes="month/day out of range")

        if _NON_STANDARD_ORDER_RE.match(text):
            return NormalizeDateOutput(
                status="ambiguous", notes="non-standard field order (MM-YYYY-DD-shaped)"
            )

        if m := _SLASH_RE.match(text):
            a, b, year_raw = (int(x) for x in m.groups())
            year = year_raw + 2000 if year_raw < 100 else year_raw
            if a > 12 and b <= 12:
                return NormalizeDateOutput(
                    normalized_date=f"{year:04d}-{b:02d}-{a:02d}", status="ok"
                )
            if b > 12 and a <= 12:
                return NormalizeDateOutput(
                    normalized_date=f"{year:04d}-{a:02d}-{b:02d}", status="ok"
                )
            return NormalizeDateOutput(
                status="ambiguous", notes="could be MM/DD or DD/MM"
            )

        if m := _ORDINAL_OF_MONTH_RE.search(text.lower()):
            ordinal_word, month_word = m.groups()
            ordinal_day = _ORDINAL_WORDS.get(ordinal_word)
            ordinal_month = _MONTHS.get(month_word)
            if ordinal_day and ordinal_month:
                return NormalizeDateOutput(
                    normalized_date=(
                        f"{_DEFAULT_SYNTHETIC_YEAR:04d}-{ordinal_month:02d}-{ordinal_day:02d}"
                    ),
                    status="ok",
                    notes=f"year assumed to be {_DEFAULT_SYNTHETIC_YEAR} (not stated)",
                )

        return NormalizeDateOutput(status="invalid", notes="unrecognized date format")


def _valid_ymd(year: int, month: int, day: int) -> bool:
    if not (1 <= month <= 12):
        return False
    days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    return 1 <= day <= days_in_month and 1900 <= year <= 2100


# --- validate_medical_field ----------------------------------------------------

_VALID_SEXES = {"male", "female"}
_DOSE_RE = re.compile(r"^\d+(\.\d+)?\s*(mg|mcg|g|ml)$|^\d+\s+tablet(s)?.*$", re.IGNORECASE)


class ValidateMedicalFieldInput(BaseModel):
    field_name: str
    value: str | None


class ValidateMedicalFieldOutput(BaseModel):
    valid: bool
    reason: str = ""


class ValidateMedicalFieldTool(BaseTool[ValidateMedicalFieldInput, ValidateMedicalFieldOutput]):
    name = "validate_medical_field"
    purpose = "Deterministic shape/range validation for one extracted medical field."
    timeout_seconds = 1.0
    max_retries = 2

    def _execute(
        self, tool_input: ValidateMedicalFieldInput, ctx: ToolContext
    ) -> ValidateMedicalFieldOutput:
        field, value = tool_input.field_name, tool_input.value
        if value is None:
            return ValidateMedicalFieldOutput(valid=True, reason="absent value is not invalid")

        if field == "patient.age":
            if not value.isdigit() or not (0 <= int(value) <= 130):
                return ValidateMedicalFieldOutput(valid=False, reason="age out of range 0-130")
        elif field == "patient.sex":
            if value.lower() not in _VALID_SEXES:
                return ValidateMedicalFieldOutput(valid=False, reason="unrecognized sex value")
        elif field == "product.dose":
            if not _DOSE_RE.match(value.strip()):
                return ValidateMedicalFieldOutput(
                    valid=False, reason="dose does not match expected shape"
                )
        return ValidateMedicalFieldOutput(valid=True)


# --- check_minimum_case_criteria ----------------------------------------------


class CheckMinimumCaseCriteriaInput(BaseModel):
    has_identifiable_patient: bool
    has_identifiable_reporter: bool
    has_suspect_product: bool
    has_adverse_event: bool


class CheckMinimumCaseCriteriaTool(
    BaseTool[CheckMinimumCaseCriteriaInput, MinimumCriteriaResult]
):
    name = "check_minimum_case_criteria"
    purpose = (
        "Deterministically evaluate the four minimum-case-criteria booleans "
        "against the exact rule (all four required) — never inferred, never "
        "invented."
    )
    timeout_seconds = 1.0
    max_retries = 2

    def _execute(
        self, tool_input: CheckMinimumCaseCriteriaInput, ctx: ToolContext
    ) -> MinimumCriteriaResult:
        missing = []
        if not tool_input.has_identifiable_patient:
            missing.append("patient")
        if not tool_input.has_identifiable_reporter:
            missing.append("reporter")
        if not tool_input.has_suspect_product:
            missing.append("suspect_product")
        if not tool_input.has_adverse_event:
            missing.append("adverse_event")
        return MinimumCriteriaResult(
            has_identifiable_patient=tool_input.has_identifiable_patient,
            has_identifiable_reporter=tool_input.has_identifiable_reporter,
            has_suspect_product=tool_input.has_suspect_product,
            has_adverse_event=tool_input.has_adverse_event,
            missing_criteria=missing,
        )


# --- detect_explicit_triage_indicators -----------------------------------

_TRIAGE_INDICATOR_PHRASES: dict[TriageIndicator, tuple[str, ...]] = {
    TriageIndicator.DEATH: ("died", "death", "deceased", "fatal"),
    TriageIndicator.LIFE_THREATENING: (
        "life-threatening", "life threatening", "anaphylactic", "anaphylaxis",
        "loss of consciousness", "cardiac arrest",
    ),
    TriageIndicator.HOSPITALIZATION: (
        "hospitalized", "hospitalised", "hospitalization", "admitted to the hospital",
        "admitted to hospital",
    ),
    TriageIndicator.DISABILITY: ("permanent disability", "incapacit"),
    TriageIndicator.CONGENITAL_ANOMALY: (
        "congenital anomaly", "birth defect", "congenital malformation",
    ),
    TriageIndicator.OTHER_MEDICALLY_IMPORTANT: (
        "medically important", "required intervention to prevent",
    ),
    TriageIndicator.SEVERE_HYPOGLYCEMIA: (
        "severe hypoglycemia", "blood sugar crashed", "unconscious from low blood sugar",
        "critically low blood sugar",
    ),
    TriageIndicator.SEVERE_INJECTION_SITE_REACTION: (
        "severe injection site reaction", "necrosis at the injection site",
        "necrosis at injection site",
    ),
}


class DetectExplicitTriageIndicatorsInput(BaseModel):
    text: str = Field(max_length=1_000_000)


class TriageIndicatorMatch(BaseModel):
    indicator: TriageIndicator
    matched_phrase: str


class DetectExplicitTriageIndicatorsOutput(BaseModel):
    matches: list[TriageIndicatorMatch] = Field(default_factory=list)

    @property
    def indicators(self) -> list[TriageIndicator]:
        return sorted({m.indicator for m in self.matches}, key=lambda i: i.value)


class DetectExplicitTriageIndicatorsTool(
    BaseTool[DetectExplicitTriageIndicatorsInput, DetectExplicitTriageIndicatorsOutput]
):
    name = "detect_explicit_triage_indicators"
    purpose = (
        "Detect ONLY explicit triage-indicator phrases via deterministic "
        "keyword matching. Never infers priority itself; the Triage Agent "
        "still requires human confirmation of anything derived from this."
    )
    timeout_seconds = 1.0
    max_retries = 2

    def _execute(
        self, tool_input: DetectExplicitTriageIndicatorsInput, ctx: ToolContext
    ) -> DetectExplicitTriageIndicatorsOutput:
        lowered = tool_input.text.lower()
        matches = [
            TriageIndicatorMatch(indicator=indicator, matched_phrase=phrase)
            for indicator, phrases in _TRIAGE_INDICATOR_PHRASES.items()
            for phrase in phrases
            if phrase in lowered
        ]
        limited, _ = limit_results(matches, 50)
        return DetectExplicitTriageIndicatorsOutput(matches=limited)


# --- suggest_triage_priority -----------------------------------------------

_CRITICAL_INDICATORS = {TriageIndicator.DEATH, TriageIndicator.LIFE_THREATENING}
_HIGH_INDICATORS = {
    TriageIndicator.HOSPITALIZATION,
    TriageIndicator.DISABILITY,
    TriageIndicator.SEVERE_HYPOGLYCEMIA,
    TriageIndicator.SEVERE_INJECTION_SITE_REACTION,
}
_MEDIUM_INDICATORS = {
    TriageIndicator.CONGENITAL_ANOMALY,
    TriageIndicator.OTHER_MEDICALLY_IMPORTANT,
}


def suggest_triage_priority(
    indicators: list[TriageIndicator],
) -> tuple[TriagePriority, str]:
    """Deterministic, documented mapping from explicit indicators to a
    SUGGESTED triage priority. This is an AI suggestion only — it is never a
    final clinical or regulatory decision, and the Triage Agent always
    routes the case for mandatory human confirmation regardless of the
    priority returned here (see CLAUDE.md Absolute Boundaries).

    Shared by ``SuggestTriagePriorityTool`` and the golden dataset generator
    so the tool being tested and the labels testing it never drift apart.
    """
    present = set(indicators)
    if present & _CRITICAL_INDICATORS:
        matched = sorted(i.value for i in present & _CRITICAL_INDICATORS)
        return TriagePriority.CRITICAL, f"Explicit indicator(s) matched: {', '.join(matched)}"
    if present & _HIGH_INDICATORS:
        matched = sorted(i.value for i in present & _HIGH_INDICATORS)
        return TriagePriority.HIGH, f"Explicit indicator(s) matched: {', '.join(matched)}"
    if present & _MEDIUM_INDICATORS:
        matched = sorted(i.value for i in present & _MEDIUM_INDICATORS)
        return TriagePriority.MEDIUM, f"Explicit indicator(s) matched: {', '.join(matched)}"
    return TriagePriority.LOW, "No explicit triage indicator phrases matched."


class SuggestTriagePriorityInput(BaseModel):
    indicators: list[TriageIndicator] = Field(default_factory=list)


class SuggestTriagePriorityOutput(BaseModel):
    suggested_priority: TriagePriority
    rationale: str


class SuggestTriagePriorityTool(BaseTool[SuggestTriagePriorityInput, SuggestTriagePriorityOutput]):
    name = "suggest_triage_priority"
    purpose = (
        "Deterministically map explicit triage indicators to a SUGGESTED "
        "priority (LOW/MEDIUM/HIGH/CRITICAL). AI suggestion only — human "
        "review is mandatory before any action is taken on a case."
    )
    timeout_seconds = 1.0
    max_retries = 2

    def _execute(
        self, tool_input: SuggestTriagePriorityInput, ctx: ToolContext
    ) -> SuggestTriagePriorityOutput:
        priority, rationale = suggest_triage_priority(tool_input.indicators)
        return SuggestTriagePriorityOutput(suggested_priority=priority, rationale=rationale)
