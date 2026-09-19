# PharmaSentry AI — Build Progress

Tracks completed and pending work against the phased plan. Update this file at the
end of every phase. Never mark an item done unless it was actually created/run and
verified (tests executed, not assumed).

Legend: `[x]` done and verified · `[~]` partially done · `[ ]` not started

This file was restructured on 2026-09-19 to follow the 15-phase plan given in
that day's build instruction (superseding the shorter Phase 3–12 placeholder
list that used to appear below Phase 2). Phase 1 and Phase 2 below reflect
what was actually built in earlier sessions; Phase 2 is being *extended* in
this session to meet the fuller spec (120+ records, more categories, more
per-record fields, 3-way split) given now.

## Phase 1 — Architecture, Scaffold, Models, State, Demo Case, Initial Tests — DONE
- [x] Repository directory structure (matches CLAUDE.md `Repository` section)
- [x] `README.md`, `.env.example`, `requirements.txt`, `Makefile`,
      `docker-compose.yml` (skeleton)
- [x] `config/` — settings loader (`config/settings.py`), `config/config.yaml`
- [x] `src/models/` — typed Pydantic domain models
- [x] `src/graph/state.py` — typed LangGraph `CaseState`
- [x] Synthetic demo case (DemoGluca) under `data/synthetic_*`
- [x] `app.py` placeholder, `tests/unit/` (22 tests)
- [x] pytest 22/22, ruff clean, mypy clean — commit `c2aaede`

## Phase 2 — Synthetic Dataset and Golden Dataset — DONE (extended)
First pass (commit `4810751`): 100 records / 8 categories / train+test split.
This session extended it to the fuller spec given in the 2026-09-19
autonomous-build instruction. Old field names (`category`,
`expected_duplicate_family`, `conflicting_fields`, `is_safety_report` as a
stored field) were renamed/replaced in place rather than duplicated,
per the "extend, don't duplicate" project rule.

- [x] Deterministic, offline generator (`scripts/generate_golden_dataset.py`),
      re-verified byte-identical across two runs via sha256 of the full
      `cases/` directory listing
- [x] **138 records** (>=120 required) across **21 case types**: complete
      (15), incomplete (15), serious (12), non_serious (12), non_safety (5),
      exact_duplicate (8, 4 families), near_duplicate (8, 4 families),
      similar_non_duplicate (6, 3 look-alike-but-not-duplicate pairs),
      conflicting (5), missing_suspect_product (4), missing_adverse_event
      (4), missing_reporter (4), missing_identifiable_patient (4), poor_ocr
      (5, deterministically corrupted attachment text), multilingual (5,
      Spanish/English), product_alias (5), event_synonym (5),
      malformed_date (5, some genuinely unresolvable by design),
      prompt_injection (5), tool_injection (3), approval_bypass_attempt (3)
- [x] Extended `GoldenCase` schema (`src/evaluation/schemas.py`):
      `case_type` (renamed from `category`), `duplicate_family_id` (renamed
      from `expected_duplicate_family`), `expected_duplicate_matches`,
      `expected_safety_classification` (3-way: safety_report/non_safety/
      uncertain, replacing a bare `is_safety_report` field — now a computed
      property), `email_subject`/`email_body` (with `email_text` as a
      computed convenience property), `attachment_metadata`
      (`AttachmentMetadata` model: has_attachment, filename, media_type,
      size_bytes, page_count, ocr_applied, ocr_quality),
      `expected_source_evidence`, `expected_conflicting_fields` (renamed
      from `conflicting_fields`), `expected_routing_decision` (reuses the
      existing `RouteReason` enum from `src/models/enums.py` rather than
      duplicating it), `prohibited_conclusions`, `expected_human_review_required`,
      `synthetic_data: Literal[True]`. `ExpectedMinimumCriteria` gained a
      `status: complete|incomplete|uncertain` field with a model validator
      enforcing consistency with the four boolean criteria.
- [x] Deterministic train/validation/test split (`family_aware_split`):
      84 train / 27 validation / 27 test. Fixed a real bug found while
      verifying: allocating train and validation independently via
      `round()` could leave a small category (e.g. the 4-family
      exact-duplicate category) with zero families in test; rewrote the
      allocator to assign test first and guarantee >=1 family in both test
      and validation whenever the category has enough families — every one
      of the 21 categories now has real representation in all three splits.
- [x] Zero duplicate-family leakage verified
      (`validate_no_family_leakage() == []`) and a dedicated test
      (`test_no_duplicate_family_straddles_splits`) proves it; every
      case_id appears in exactly one of train/validation/test
      (`test_split_covers_every_case_exactly_once`)
- [x] `scripts/seed_synthetic_data.py` rewritten as a real executable: runs
      the generator, then validates record counts, unique case IDs, split
      isolation, label completeness, evidence validity (every
      `expected_source_evidence` snippet is checked to actually appear in
      that case's own email/attachment text — not just asserted), synthetic
      markers, and duplicate-family consistency. All seven checks pass.
- [x] Found and fixed a second real bug during validation: OCR corruption
      for `poor_ocr` cases could occasionally mangle the
      "[SYNTHETIC / FICTIONAL DOCUMENT ...]" marker itself; the corruption
      routine now leaves the marker line untouched.
- [x] `tests/unit/test_evaluation_schemas.py`,
      `tests/integration/test_golden_dataset_integrity.py` (23 tests,
      rewritten for the new schema/counts + new checks for the new
      categories), `tests/integration/test_generate_golden_dataset.py`,
      `tests/integration/test_run_golden_evaluation.py` updated for the new
      field names/counts
- [x] pytest **71/71 passed**, ruff clean, mypy clean (35 files) after the
      extension (commands and exact output in the session transcript)

## Notes / deviations logged during Phase 2 extension
- CLAUDE.md's Hybrid RAG section calls for MRR too; not part of this
  message's explicit metric list but already implemented
  (`mean_reciprocal_rank` in `src/evaluation/metrics.py`) from the prior
  session, kept for Phase 5.
- `expected_human_review_required` marks cases needing **escalated/early**
  human attention (seriousness, duplicates, conflicts, incomplete minimum
  criteria, injection/bypass attempts) — not the same as "will a human
  approve this case eventually," which is true for every case per the
  Absolute Boundaries (final review is mandatory regardless of this flag).
- `similar_non_duplicate` cases are each their own singleton family
  (`expected_duplicate_matches == []`); the two members of a pair are
  cross-referenced only in free-text `notes`, since they are, by design,
  NOT a duplicate-family match.
- Old `evaluations/results/phase2_baseline_metrics.json` numbers from the
  100-case dataset are stale and were regenerated against the 138-case
  dataset; the file itself is gitignored (regenerated by `make eval`, never
  committed).

## Phase 3 — Deterministic Tool Layer — [ ]
## Phase 4 — Memory — [ ]
## Phase 5 — Hybrid RAG — [ ]
## Phase 6 — LLM Provider Abstraction — [ ]
## Phase 7 — Agents — [ ]
## Phase 8 — LangGraph Orchestration — [ ]
## Phase 9 — Evaluation Framework — [ ]
## Phase 10 — Observability and Traceability — [ ]
## Phase 11 — Security, Guardrails, and Red Teaming — [ ]
## Phase 12 — Streamlit User Interface — [ ]
## Phase 13 — API and Service Layer — [ ]
## Phase 14 — CI/CD and Continuous Evaluation — [ ]
## Phase 15 — Documentation — [ ]
## Final Verification — [ ]

(Each phase above gets its own detailed section, filled in as completed —
see below. Sections are appended in order; nothing is marked `[x]` above
until its detailed section below has actually been executed and verified.)
