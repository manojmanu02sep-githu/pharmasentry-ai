"""Shared synthetic reference data: fictional diabetes-injectable product
aliases and event term synonyms.

Single source of truth for this data — both the domain lookup tools
(`lookup_product_alias`, `lookup_event_term`) and the golden dataset
generator (`scripts/generate_golden_dataset.py`) import from here, so the
tool being tested and the fixtures testing it can never silently drift
apart. This will become the seed data for semantic memory in Phase 4;
until then, it lives here as the tools' own reference data.

All products are FICTIONAL diabetes-injectable drugs only (this system's
business use case is Diabetes Injection Safety Email Triage — see
CLAUDE.md). No real drug names, dosing, or manufacturers are referenced.
"""

from __future__ import annotations

CANONICAL_PRODUCTS: tuple[str, ...] = ("DemoInsulex", "DemoBasalin", "DemoGlutide")

PRODUCT_ALIASES: dict[str, str] = {
    "DemoInsulex Flex": "DemoInsulex",
    "Demo-Insulex": "DemoInsulex",
    "DemoBasalin Pen": "DemoBasalin",
    "Demo Basalin": "DemoBasalin",
    "DemoGlutide Injection": "DemoGlutide",
    "Demo-Glutide": "DemoGlutide",
    "DemoGluca": "DemoGlutide",
    "Demo-Gluca": "DemoGlutide",
}

EVENT_SYNONYMS: dict[str, str] = {
    "throwing up repeatedly": "vomiting",
    "a tummy ache": "abdominal pain",
    "passed out": "loss of consciousness",
    "a pounding headache": "a severe headache",
    "puffy face and lips": "facial swelling",
    "sugar crashed hard": "severe hypoglycemia",
    "a burning red spot where I injected": "injection site reaction",
}
