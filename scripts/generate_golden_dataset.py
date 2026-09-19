#!/usr/bin/env python3
"""Generate the synthetic pharmacovigilance golden dataset.

Deterministic and fully offline: no LLM calls, no network access, no real
patient/reporter/company data (see data/SYNTHETIC_DATA_NOTICE.md). Re-running
with the same --seed produces byte-identical output, which is what makes this
a *golden* dataset rather than a one-off sample.

Distribution (100 cases total):
    20 complete, 20 incomplete, 15 serious, 15 non-serious,
    10 exact duplicates (5 families x 2), 10 near duplicates (5 families x 2),
    5 conflicting email-vs-attachment, 5 non-safety emails.

Every case's email_text/attachment_text is rendered FROM the same fact
dict used to derive its expected_* labels, so the dataset is internally
consistent by construction (the labels are not hand-typed separately from
the text).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.schemas import (  # noqa: E402
    DatasetManifest,
    DatasetSplit,
    ExpectedMinimumCriteria,
    GoldenCase,
    GoldenCaseCategory,
)

GOLDEN_DIR = REPO_ROOT / "evaluations" / "golden_dataset"
CASES_DIR = GOLDEN_DIR / "cases"

DISTRIBUTION: dict[GoldenCaseCategory, int] = {
    GoldenCaseCategory.COMPLETE: 20,
    GoldenCaseCategory.INCOMPLETE: 20,
    GoldenCaseCategory.SERIOUS: 15,
    GoldenCaseCategory.NON_SERIOUS: 15,
    GoldenCaseCategory.EXACT_DUPLICATE: 10,
    GoldenCaseCategory.NEAR_DUPLICATE: 10,
    GoldenCaseCategory.CONFLICTING: 5,
    GoldenCaseCategory.NON_SAFETY: 5,
}
assert sum(DISTRIBUTION.values()) == 100, "distribution must total 100 cases"

TEST_FRACTION = 0.2

# --- Fictional data pools (no real people, products, or organizations) ---

PRODUCTS = [
    ("DemoGluca", "type 2 diabetes"),
    ("DemoCardolol", "hypertension"),
    ("DemoZanix", "generalized anxiety disorder"),
]
DOSES = ["5mg", "10mg", "25mg", "50mg", "500mg", "1 tablet twice daily"]
ROUTES = ["oral", "subcutaneous injection"]
SEXES = ["male", "female"]
CLINICS = [
    "Fictional Community Clinic",
    "Demo Regional Hospital",
    "Fictional General Hospital (demo)",
    "Demo Family Practice",
]
REPORTER_ROLES = [
    "nurse practitioner",
    "staff pharmacist",
    "treating physician",
    "clinical care coordinator",
]
REPORTER_FIRST = ["R.", "A.", "J.", "M.", "S.", "K.", "T.", "L.", "P.", "D."]
REPORTER_LAST = [
    "Tanaka", "Okafor", "Silva", "Novak", "Haddad",
    "Larsen", "Petrov", "Reyes", "Kim", "Duval",
]

# (event_description, seriousness_indicator, discharge_diagnosis)
SERIOUS_EVENTS = [
    ("severe abdominal pain and vomiting", "hospitalization", "acute pancreatitis"),
    ("sudden difficulty breathing and facial swelling", "life_threatening",
     "anaphylactic reaction"),
    ("chest pain followed by loss of consciousness", "life_threatening", "cardiac arrhythmia"),
    ("high fever and confusion", "hospitalization", "severe infection"),
    ("severe abdominal pain radiating to the back", "hospitalization", "acute pancreatitis"),
]
NON_SERIOUS_EVENTS = [
    "mild nausea", "a mild headache", "occasional dizziness",
    "a mild skin rash", "mild fatigue",
]
OUTCOMES = ["stable, improving", "fully recovered", "recovering, still under observation"]

# Any event (serious or not), each paired with its indicator/diagnosis (None
# for non-serious events) — used where a duplicate pair may or may not be
# a serious case.
ANY_EVENT_WITH_INDICATOR: list[tuple[str, str | None, str | None]] = [
    *SERIOUS_EVENTS,
    *[(e, None, None) for e in NON_SERIOUS_EVENTS],
]

NON_SAFETY_EMAILS = [
    (
        "Question about DemoGluca packaging",
        "I noticed the box design for DemoGluca changed recently and wanted "
        "to confirm it's still the same product before my next refill.",
    ),
    (
        "Request for product literature",
        "Could you send the current prescribing information for "
        "DemoCardolol? I'd like to review it before our next appointment.",
    ),
    (
        "Billing question about a recent refill",
        "I believe I was double-charged for a DemoZanix refill last month "
        "and would like this looked into and corrected.",
    ),
    (
        "General question about tablet appearance",
        "The DemoGluca tablets from my last refill look slightly different "
        "in color from the previous batch. Is that expected?",
    ),
    (
        "Request for insurance coverage documentation",
        "Can you provide documentation of DemoCardolol coverage that I can "
        "forward to my insurer?",
    ),
]

MISSABLE_FIELDS = ["product.dose", "product.treatment_start_date", "product.route"]


def rng_for(seed: int, index: int) -> random.Random:
    return random.Random(seed * 100_003 + index)


def pick_reporter(rng: random.Random) -> tuple[str, str, str]:
    name = f"{rng.choice(REPORTER_FIRST)} {rng.choice(REPORTER_LAST)}"
    role = rng.choice(REPORTER_ROLES)
    clinic = rng.choice(CLINICS)
    return name, role, clinic


def base_facts(rng: random.Random) -> dict[str, str]:
    product, context = rng.choice(PRODUCTS)
    reporter_name, reporter_role, clinic = pick_reporter(rng)
    return {
        "age": str(rng.randint(19, 88)),
        "sex": rng.choice(SEXES),
        "product": product,
        "product_context": context,
        "dose": rng.choice(DOSES),
        "route": rng.choice(ROUTES),
        "dose_number": rng.choice(["first", "second", "third", "fourth"]),
        "onset_days": str(rng.randint(1, 10)),
        "treatment_start_date": f"2026-{rng.randint(1, 8):02d}-{rng.randint(1, 28):02d}",
        "reporter_name": reporter_name,
        "reporter_role": reporter_role,
        "clinic": clinic,
    }


def email_header(case_num: int) -> str:
    return (
        "[SYNTHETIC / FICTIONAL DATA — educational prototype only, no "
        "real patient or reporter information]\n\n"
        "Hello Safety Team,\n\n"
    )


def _stable_id(text: str) -> int:
    """Deterministic small integer derived from text.

    Deliberately not Python's built-in ``hash()``: str hashing is
    process-randomized (PYTHONHASHSEED) unless disabled, which would break
    this generator's "same --seed -> byte-identical output" guarantee.
    """
    return sum(ord(c) for c in text) % 10_000


def email_signoff(facts: dict[str, str]) -> str:
    return (
        f"\nRegards,\n{facts['reporter_name']} (fictional reporter)\n"
        f"{facts['clinic']}\n"
        f"Contact: demo.reporter{_stable_id(facts['reporter_name'])}"
        f"@fictionalclinic-demo.example (synthetic contact info)\n"
    )


def render_email(
    facts: dict[str, str],
    event_description: str,
    *,
    mention_dose: bool = True,
    mention_start_date: bool = True,
    mention_route: bool = True,
    hospitalized: bool = False,
    resend_note: str | None = None,
    reword: bool = False,
    override_dose: str | None = None,
    override_onset_days: str | None = None,
) -> str:
    lines = [email_header(0)]
    lines.append(
        f"I am {'a ' if facts['reporter_role'][0] not in 'aeiou' else 'an '}"
        f"{facts['reporter_role']} at {facts['clinic']} (a fabricated "
        "organization) and I would like to report a possible adverse "
        "event." if not resend_note else resend_note
    )
    lines.append("")

    dose_phrase = ""
    dose_value = override_dose if override_dose is not None else facts["dose"]
    if mention_dose:
        dose_phrase = f" (dose: {dose_value})" if not reword else f", at a dose of {dose_value},"
    route_phrase = f" via {facts['route']}" if mention_route else ""

    intro = (
        f"A {facts['age']}-year-old {facts['sex']} patient with a history of "
        f"{facts['product_context']} received their {facts['dose_number']} "
        f"dose of {facts['product']}{dose_phrase}{route_phrase}."
        if not reword
        else (
            f"Our patient, a {facts['sex']} in their {int(facts['age']) // 10 * 10}s "
            f"with {facts['product_context']}, was given {facts['product']}"
            f"{dose_phrase} ({facts['dose_number']} dose)."
        )
    )
    lines.append(intro)

    if mention_start_date:
        lines.append(f"Treatment started on {facts['treatment_start_date']}.")

    onset_days = override_onset_days if override_onset_days is not None else facts["onset_days"]
    lines.append(
        f"Approximately {onset_days} day(s) after that dose, the patient "
        f"developed {event_description}."
        if not reword
        else f"About {onset_days} days later, they developed {event_description}."
    )

    if hospitalized:
        lines.append("The patient was hospitalized as a result.")

    lines.append("")
    lines.append("Please let me know if you need anything else.")
    lines.append(email_signoff(facts))

    subject_event = event_description[:60]
    subject = (
        f"Subject: Possible adverse event report - {facts['product']} "
        f"patient with {subject_event}"
    )
    return subject + "\n\n" + "\n".join(lines)


def render_attachment(
    facts: dict[str, str],
    discharge_diagnosis: str,
    outcome_description: str,
    *,
    admitting_complaint: str,
    override_dose: str | None = None,
    override_start_date: str | None = None,
) -> str:
    dose_value = override_dose if override_dose is not None else facts["dose"]
    start_date = (
        override_start_date
        if override_start_date is not None
        else facts["treatment_start_date"]
    )
    return (
        "[SYNTHETIC / FICTIONAL DOCUMENT — educational prototype only]\n"
        f"{facts['clinic'].upper()} (demo) - DISCHARGE SUMMARY\n"
        "Page 1 of 1\n\n"
        f"Patient: [synthetic patient, {facts['age']}-year-old {facts['sex']}] "
        "(no real identity - demo only)\n"
        f"Admitting complaint: {admitting_complaint}\n"
        f"Relevant history: {facts['product_context']}\n"
        f"Suspect product: {facts['product']}, dose {dose_value}, "
        f"started {start_date}\n\n"
        "Course of care:\n"
        f"Patient was evaluated and treated. Findings were consistent with "
        f"{discharge_diagnosis}.\n\n"
        f"Discharge diagnosis: {discharge_diagnosis} (fictional case).\n"
        f"Outcome at discharge: {outcome_description}.\n\n"
        "-- End of synthetic discharge summary --\n"
    )


def full_minimum_criteria() -> ExpectedMinimumCriteria:
    return ExpectedMinimumCriteria(
        has_identifiable_patient=True,
        has_identifiable_reporter=True,
        has_suspect_product=True,
        has_adverse_event=True,
        missing_criteria=[],
    )


def narrative_facts(
    facts: dict[str, str],
    event_description: str,
    *,
    hospitalized: bool,
    discharge_diagnosis: str | None,
    outcome_description: str | None,
    mention_dose: bool = True,
    mention_start_date: bool = True,
) -> list[str]:
    out = [
        f"{facts['age']}-year-old {facts['sex']} patient with a history of "
        f"{facts['product_context']}.",
        f"Received {facts['dose_number']} dose of {facts['product']}"
        + (f" ({facts['dose']})" if mention_dose else "") + ".",
    ]
    if mention_start_date:
        out.append(f"Treatment started on {facts['treatment_start_date']}.")
    out.append(
        f"Approximately {facts['onset_days']} day(s) after that dose, "
        f"developed {event_description}."
    )
    if hospitalized:
        out.append("Patient was hospitalized.")
    if discharge_diagnosis:
        out.append(f"Discharge diagnosis: {discharge_diagnosis}.")
    if outcome_description:
        out.append(f"Outcome: {outcome_description}.")
    return out


# Distinct offset ranges per category so `rng_for(seed, offset + idx)` never
# collides across categories (which would otherwise give unrelated cases at
# the same index identical reporter/patient/product facts).
_OFFSET_COMPLETE = 0
_OFFSET_INCOMPLETE = 100
_OFFSET_SERIOUS = 200
_OFFSET_NON_SERIOUS = 300
_OFFSET_EXACT_DUP = 1000
_OFFSET_NEAR_DUP = 2000
_OFFSET_CONFLICTING = 3000
_OFFSET_NON_SAFETY = 4000


def build_complete(idx: int, seed: int) -> GoldenCase:
    rng = rng_for(seed, _OFFSET_COMPLETE + idx)
    facts = base_facts(rng)
    event = rng.choice(NON_SERIOUS_EVENTS)
    outcome = rng.choice(OUTCOMES)
    case_id = f"complete_{idx:03d}"

    email = render_email(facts, event, hospitalized=False)
    attachment = render_attachment(
        facts, discharge_diagnosis=f"observation for {event}", outcome_description=outcome,
        admitting_complaint=event,
    )

    expected_fields: dict[str, str | None] = {
        "patient.age": facts["age"],
        "patient.sex": facts["sex"],
        "reporter.reporter_type": facts["reporter_role"],
        "reporter.name": facts["reporter_name"],
        "product.product_name": facts["product"],
        "product.dose": facts["dose"],
        "product.route": facts["route"],
        "product.treatment_start_date": facts["treatment_start_date"],
        "event.event_description": event,
        "event.event_onset_date": f"{facts['onset_days']} day(s) after dose",
        "outcome.outcome_description": outcome,
        "outcome.hospitalized": "false",
    }
    return GoldenCase(
        case_id=case_id,
        category=GoldenCaseCategory.COMPLETE,
        is_safety_report=True,
        email_text=email,
        attachment_text=attachment,
        expected_extracted_fields=expected_fields,
        expected_minimum_criteria=full_minimum_criteria(),
        expected_seriousness_indicators=[],
        expected_missing_fields=[],
        expected_duplicate_family=f"fam_{case_id}",
        expected_narrative_facts=narrative_facts(
            facts, event, hospitalized=False,
            discharge_diagnosis=None, outcome_description=outcome,
        ),
        notes="Fully detailed, non-serious case.",
    )


def build_incomplete(idx: int, seed: int) -> GoldenCase:
    rng = rng_for(seed, _OFFSET_INCOMPLETE + idx)
    facts = base_facts(rng)
    event = rng.choice(NON_SERIOUS_EVENTS + [e for e, _, _ in SERIOUS_EVENTS])
    is_serious = event in [e for e, _, _ in SERIOUS_EVENTS]
    seriousness = next((s for e, s, _ in SERIOUS_EVENTS if e == event), None)
    diagnosis = next((d for e, _, d in SERIOUS_EVENTS if e == event), None)
    case_id = f"incomplete_{idx:03d}"

    n_missing = rng.choice([1, 2])
    missing = rng.sample(MISSABLE_FIELDS, n_missing)
    mention_dose = "product.dose" not in missing
    mention_start_date = "product.treatment_start_date" not in missing
    mention_route = "product.route" not in missing

    outcome = rng.choice(OUTCOMES)
    email = render_email(
        facts, event,
        mention_dose=mention_dose, mention_start_date=mention_start_date,
        mention_route=mention_route, hospitalized=is_serious,
    )
    attachment = render_attachment(
        facts, discharge_diagnosis=diagnosis or f"observation for {event}",
        outcome_description=outcome, admitting_complaint=event,
        override_dose=facts["dose"] if mention_dose else "not documented",
        override_start_date=(
            facts["treatment_start_date"] if mention_start_date else "not documented"
        ),
    )

    expected_fields: dict[str, str | None] = {
        "patient.age": facts["age"],
        "patient.sex": facts["sex"],
        "reporter.reporter_type": facts["reporter_role"],
        "reporter.name": facts["reporter_name"],
        "product.product_name": facts["product"],
        "product.dose": facts["dose"] if mention_dose else None,
        "product.route": facts["route"] if mention_route else None,
        "product.treatment_start_date": (
            facts["treatment_start_date"] if mention_start_date else None
        ),
        "event.event_description": event,
        "event.event_onset_date": f"{facts['onset_days']} day(s) after dose",
        "outcome.outcome_description": outcome,
        "outcome.hospitalized": "true" if is_serious else "false",
    }
    indicators = [seriousness] if seriousness else []
    if is_serious:
        indicators = sorted(set(indicators + ["hospitalization"]))

    return GoldenCase(
        case_id=case_id,
        category=GoldenCaseCategory.INCOMPLETE,
        is_safety_report=True,
        email_text=email,
        attachment_text=attachment,
        expected_extracted_fields=expected_fields,
        expected_minimum_criteria=full_minimum_criteria(),
        expected_seriousness_indicators=indicators,
        expected_missing_fields=sorted(missing),
        expected_duplicate_family=f"fam_{case_id}",
        expected_narrative_facts=narrative_facts(
            facts, event, hospitalized=is_serious,
            discharge_diagnosis=diagnosis, outcome_description=outcome,
            mention_dose=mention_dose, mention_start_date=mention_start_date,
        ),
        notes=f"Missing fields (not stated in email or attachment): {sorted(missing)}.",
    )


def build_serious(idx: int, seed: int) -> GoldenCase:
    rng = rng_for(seed, _OFFSET_SERIOUS + idx)
    facts = base_facts(rng)
    event, indicator, diagnosis = rng.choice(SERIOUS_EVENTS)
    outcome = rng.choice(OUTCOMES)
    case_id = f"serious_{idx:03d}"

    email = render_email(facts, event, hospitalized=True)
    attachment = render_attachment(
        facts, discharge_diagnosis=diagnosis, outcome_description=outcome,
        admitting_complaint=event,
    )
    indicators = sorted({indicator, "hospitalization"})

    expected_fields: dict[str, str | None] = {
        "patient.age": facts["age"],
        "patient.sex": facts["sex"],
        "reporter.reporter_type": facts["reporter_role"],
        "reporter.name": facts["reporter_name"],
        "product.product_name": facts["product"],
        "product.dose": facts["dose"],
        "product.route": facts["route"],
        "product.treatment_start_date": facts["treatment_start_date"],
        "event.event_description": event,
        "event.event_onset_date": f"{facts['onset_days']} day(s) after dose",
        "outcome.outcome_description": outcome,
        "outcome.hospitalized": "true",
    }
    return GoldenCase(
        case_id=case_id,
        category=GoldenCaseCategory.SERIOUS,
        is_safety_report=True,
        email_text=email,
        attachment_text=attachment,
        expected_extracted_fields=expected_fields,
        expected_minimum_criteria=full_minimum_criteria(),
        expected_seriousness_indicators=indicators,
        expected_missing_fields=[],
        expected_duplicate_family=f"fam_{case_id}",
        expected_narrative_facts=narrative_facts(
            facts, event, hospitalized=True,
            discharge_diagnosis=diagnosis, outcome_description=outcome,
        ),
        notes="Fully detailed case with an explicit seriousness indicator.",
    )


def build_non_serious(idx: int, seed: int) -> GoldenCase:
    rng = rng_for(seed, _OFFSET_NON_SERIOUS + idx)
    facts = base_facts(rng)
    event = rng.choice(NON_SERIOUS_EVENTS)
    outcome = rng.choice(OUTCOMES)
    case_id = f"non_serious_{idx:03d}"

    email = render_email(facts, event, hospitalized=False)
    attachment = render_attachment(
        facts, discharge_diagnosis=f"clinical review for {event}",
        outcome_description=outcome, admitting_complaint=event,
    )

    expected_fields: dict[str, str | None] = {
        "patient.age": facts["age"],
        "patient.sex": facts["sex"],
        "reporter.reporter_type": facts["reporter_role"],
        "reporter.name": facts["reporter_name"],
        "product.product_name": facts["product"],
        "product.dose": facts["dose"],
        "product.route": facts["route"],
        "product.treatment_start_date": facts["treatment_start_date"],
        "event.event_description": event,
        "event.event_onset_date": f"{facts['onset_days']} day(s) after dose",
        "outcome.outcome_description": outcome,
        "outcome.hospitalized": "false",
    }
    return GoldenCase(
        case_id=case_id,
        category=GoldenCaseCategory.NON_SERIOUS,
        is_safety_report=True,
        email_text=email,
        attachment_text=attachment,
        expected_extracted_fields=expected_fields,
        expected_minimum_criteria=full_minimum_criteria(),
        expected_seriousness_indicators=[],
        expected_missing_fields=[],
        expected_duplicate_family=f"fam_{case_id}",
        expected_narrative_facts=narrative_facts(
            facts, event, hospitalized=False,
            discharge_diagnosis=None, outcome_description=outcome,
        ),
        notes="Fully detailed case with no seriousness indicator.",
    )


def build_exact_duplicate_pair(pair_idx: int, seed: int) -> list[GoldenCase]:
    rng = rng_for(seed, _OFFSET_EXACT_DUP + pair_idx)
    facts = base_facts(rng)
    event, indicator, diagnosis = rng.choice(ANY_EVENT_WITH_INDICATOR)
    is_serious = indicator is not None
    outcome = rng.choice(OUTCOMES)
    family = f"fam_exact_{pair_idx:02d}"

    cases = []
    resend_note = "This is a resend of my earlier report in case it did not arrive."
    for suffix, resend in (("a", None), ("b", resend_note)):
        case_id = f"exact_duplicate_{pair_idx:02d}_{suffix}"
        email = render_email(facts, event, hospitalized=is_serious, resend_note=resend)
        attachment = render_attachment(
            facts, discharge_diagnosis=diagnosis or f"observation for {event}",
            outcome_description=outcome, admitting_complaint=event,
        )
        expected_fields: dict[str, str | None] = {
            "patient.age": facts["age"],
            "patient.sex": facts["sex"],
            "reporter.reporter_type": facts["reporter_role"],
            "reporter.name": facts["reporter_name"],
            "product.product_name": facts["product"],
            "product.dose": facts["dose"],
            "product.route": facts["route"],
            "product.treatment_start_date": facts["treatment_start_date"],
            "event.event_description": event,
            "event.event_onset_date": f"{facts['onset_days']} day(s) after dose",
            "outcome.outcome_description": outcome,
            "outcome.hospitalized": "true" if is_serious else "false",
        }
        if is_serious:
            assert indicator is not None
            indicators = sorted({indicator, "hospitalization"})
        else:
            indicators = []
        cases.append(
            GoldenCase(
                case_id=case_id,
                category=GoldenCaseCategory.EXACT_DUPLICATE,
                is_safety_report=True,
                email_text=email,
                attachment_text=attachment,
                expected_extracted_fields=expected_fields,
                expected_minimum_criteria=full_minimum_criteria(),
                expected_seriousness_indicators=indicators,
                expected_missing_fields=[],
                expected_duplicate_family=family,
                expected_narrative_facts=narrative_facts(
                    facts, event, hospitalized=is_serious,
                    discharge_diagnosis=diagnosis, outcome_description=outcome,
                ),
                notes=f"Exact duplicate family {family}, member {suffix}: identical "
                "underlying facts, resent/re-transcribed.",
            )
        )
    return cases


def build_near_duplicate_pair(pair_idx: int, seed: int) -> list[GoldenCase]:
    rng = rng_for(seed, _OFFSET_NEAR_DUP + pair_idx)
    facts = base_facts(rng)
    event, indicator, diagnosis = rng.choice(ANY_EVENT_WITH_INDICATOR)
    is_serious = indicator is not None
    outcome = rng.choice(OUTCOMES)
    family = f"fam_near_{pair_idx:02d}"

    cases = []
    # Member "a": dose omitted, plain wording. Member "b": dose present, reworded.
    for suffix, mention_dose, reword in (("a", False, False), ("b", True, True)):
        case_id = f"near_duplicate_{pair_idx:02d}_{suffix}"
        email = render_email(
            facts, event, mention_dose=mention_dose, hospitalized=is_serious, reword=reword,
        )
        attachment = render_attachment(
            facts, discharge_diagnosis=diagnosis or f"observation for {event}",
            outcome_description=outcome, admitting_complaint=event,
            override_dose=facts["dose"] if mention_dose else "not documented",
        )
        expected_fields: dict[str, str | None] = {
            "patient.age": facts["age"],
            "patient.sex": facts["sex"],
            "reporter.reporter_type": facts["reporter_role"],
            "reporter.name": facts["reporter_name"],
            "product.product_name": facts["product"],
            "product.dose": facts["dose"] if mention_dose else None,
            "product.route": facts["route"],
            "product.treatment_start_date": facts["treatment_start_date"],
            "event.event_description": event,
            "event.event_onset_date": f"{facts['onset_days']} day(s) after dose",
            "outcome.outcome_description": outcome,
            "outcome.hospitalized": "true" if is_serious else "false",
        }
        if is_serious:
            assert indicator is not None
            indicators = sorted({indicator, "hospitalization"})
        else:
            indicators = []
        cases.append(
            GoldenCase(
                case_id=case_id,
                category=GoldenCaseCategory.NEAR_DUPLICATE,
                is_safety_report=True,
                email_text=email,
                attachment_text=attachment,
                expected_extracted_fields=expected_fields,
                expected_minimum_criteria=full_minimum_criteria(),
                expected_seriousness_indicators=indicators,
                expected_missing_fields=[] if mention_dose else ["product.dose"],
                expected_duplicate_family=family,
                expected_narrative_facts=narrative_facts(
                    facts, event, hospitalized=is_serious,
                    discharge_diagnosis=diagnosis, outcome_description=outcome,
                    mention_dose=mention_dose,
                ),
                notes=f"Near-duplicate family {family}, member {suffix}: same "
                "underlying case, reworded and/or missing a secondary field.",
            )
        )
    return cases


def build_conflicting(idx: int, seed: int) -> GoldenCase:
    rng = rng_for(seed, _OFFSET_CONFLICTING + idx)
    facts = base_facts(rng)
    event = rng.choice(NON_SERIOUS_EVENTS)
    outcome = rng.choice(OUTCOMES)
    case_id = f"conflicting_{idx:03d}"

    conflicting_dose = rng.choice([d for d in DOSES if d != facts["dose"]])

    email = render_email(facts, event, hospitalized=False)
    attachment = render_attachment(
        facts, discharge_diagnosis=f"clinical review for {event}",
        outcome_description=outcome, admitting_complaint=event,
        override_dose=conflicting_dose,
    )

    expected_fields: dict[str, str | None] = {
        "patient.age": facts["age"],
        "patient.sex": facts["sex"],
        "reporter.reporter_type": facts["reporter_role"],
        "reporter.name": facts["reporter_name"],
        "product.product_name": facts["product"],
        "product.dose": None,  # conflicting sources: no single ground truth
        "product.route": facts["route"],
        "product.treatment_start_date": facts["treatment_start_date"],
        "event.event_description": event,
        "event.event_onset_date": f"{facts['onset_days']} day(s) after dose",
        "outcome.outcome_description": outcome,
        "outcome.hospitalized": "false",
    }
    return GoldenCase(
        case_id=case_id,
        category=GoldenCaseCategory.CONFLICTING,
        is_safety_report=True,
        email_text=email,
        attachment_text=attachment,
        expected_extracted_fields=expected_fields,
        expected_minimum_criteria=full_minimum_criteria(),
        expected_seriousness_indicators=[],
        expected_missing_fields=[],
        expected_duplicate_family=f"fam_{case_id}",
        expected_narrative_facts=narrative_facts(
            facts, event, hospitalized=False,
            discharge_diagnosis=None, outcome_description=outcome,
        ),
        conflicting_fields=["product.dose"],
        notes=(
            f"Email states dose {facts['dose']!r}; attachment states "
            f"{conflicting_dose!r} for the same patient/product/event."
        ),
    )


def build_non_safety(idx: int, seed: int) -> GoldenCase:
    rng = rng_for(seed, _OFFSET_NON_SAFETY + idx)
    reporter_name, reporter_role, clinic = pick_reporter(rng)
    subject, body = rng.choice(NON_SAFETY_EMAILS)
    case_id = f"non_safety_{idx:03d}"

    email = (
        f"Subject: {subject}\n\n"
        "[SYNTHETIC / FICTIONAL DATA — educational prototype only]\n\n"
        "Hello,\n\n"
        f"{body}\n\n"
        "Thank you,\n"
        f"{reporter_name}\n{clinic} (fictional)\n"
    )

    return GoldenCase(
        case_id=case_id,
        category=GoldenCaseCategory.NON_SAFETY,
        is_safety_report=False,
        email_text=email,
        attachment_text="",
        expected_extracted_fields={},
        expected_minimum_criteria=ExpectedMinimumCriteria(
            has_identifiable_patient=False,
            has_identifiable_reporter=True,
            has_suspect_product=False,
            has_adverse_event=False,
            missing_criteria=["patient", "suspect_product", "adverse_event"],
        ),
        expected_seriousness_indicators=[],
        expected_missing_fields=[],
        expected_duplicate_family=f"fam_{case_id}",
        expected_narrative_facts=[],
        notes="Non-safety inquiry; no case narrative is expected.",
    )


def generate_all(seed: int) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for i in range(1, DISTRIBUTION[GoldenCaseCategory.COMPLETE] + 1):
        cases.append(build_complete(i, seed))
    for i in range(1, DISTRIBUTION[GoldenCaseCategory.INCOMPLETE] + 1):
        cases.append(build_incomplete(i, seed))
    for i in range(1, DISTRIBUTION[GoldenCaseCategory.SERIOUS] + 1):
        cases.append(build_serious(i, seed))
    for i in range(1, DISTRIBUTION[GoldenCaseCategory.NON_SERIOUS] + 1):
        cases.append(build_non_serious(i, seed))
    for i in range(1, DISTRIBUTION[GoldenCaseCategory.EXACT_DUPLICATE] // 2 + 1):
        cases.extend(build_exact_duplicate_pair(i, seed))
    for i in range(1, DISTRIBUTION[GoldenCaseCategory.NEAR_DUPLICATE] // 2 + 1):
        cases.extend(build_near_duplicate_pair(i, seed))
    for i in range(1, DISTRIBUTION[GoldenCaseCategory.CONFLICTING] + 1):
        cases.append(build_conflicting(i, seed))
    for i in range(1, DISTRIBUTION[GoldenCaseCategory.NON_SAFETY] + 1):
        cases.append(build_non_safety(i, seed))
    return cases


def family_aware_split(cases: list[GoldenCase], test_fraction: float) -> DatasetSplit:
    """Stratify per category, holding whole duplicate families together.

    Within each category, families are ordered by first appearance and the
    last ceil(test_fraction * n_families) go to test — deterministic given
    the (deterministic) generation order.
    """
    train: list[str] = []
    test: list[str] = []

    by_category: dict[GoldenCaseCategory, list[GoldenCase]] = {}
    for case in cases:
        by_category.setdefault(case.category, []).append(case)

    for category_cases in by_category.values():
        family_order: list[str] = []
        seen = set()
        for case in category_cases:
            if case.expected_duplicate_family not in seen:
                family_order.append(case.expected_duplicate_family)
                seen.add(case.expected_duplicate_family)

        n_test_families = max(1, round(len(family_order) * test_fraction))
        test_families = set(family_order[-n_test_families:])

        for case in category_cases:
            if case.expected_duplicate_family in test_families:
                test.append(case.case_id)
            else:
                train.append(case.case_id)

    return DatasetSplit(train=sorted(train), test=sorted(test))


def write_dataset(cases: list[GoldenCase], split: DatasetSplit, seed: int) -> None:
    CASES_DIR.mkdir(parents=True, exist_ok=True)

    for case in cases:
        path = CASES_DIR / f"{case.case_id}.json"
        path.write_text(case.model_dump_json(indent=2) + "\n", encoding="utf-8")

    category_counts: dict[str, int] = {}
    for case in cases:
        category_counts[case.category.value] = category_counts.get(case.category.value, 0) + 1

    manifest = DatasetManifest(
        seed=seed,
        generated_at=datetime.now(UTC),
        total_cases=len(cases),
        category_counts=category_counts,
    )
    (GOLDEN_DIR / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (GOLDEN_DIR / "split.json").write_text(
        split.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args()

    cases = generate_all(args.seed)
    if len(cases) != 100:
        print(f"Expected 100 cases, generated {len(cases)}.", file=sys.stderr)
        return 1

    case_ids = [c.case_id for c in cases]
    if len(set(case_ids)) != len(case_ids):
        print("Duplicate case_id values were generated.", file=sys.stderr)
        return 1

    split = family_aware_split(cases, TEST_FRACTION)
    write_dataset(cases, split, args.seed)

    print(f"Wrote {len(cases)} golden cases to {CASES_DIR}")
    print(f"Split: {len(split.train)} train / {len(split.test)} test")
    print(json.dumps({"category_counts": {
        cat.value: sum(1 for c in cases if c.category == cat) for cat in DISTRIBUTION
    }}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
