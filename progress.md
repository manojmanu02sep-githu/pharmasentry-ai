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

## Business-use-case pivot (2026-09-19): Diabetes Injection Safety Email Triage
Pivoted the running scenario from the original pharmacovigilance/DemoGluca
prototype to a Diabetes Injection Safety Email Triage System (doctors send
Subject + Email Body + PDF/DOCX attachments; agents triage LOW/MEDIUM/HIGH/
CRITICAL as an AI *suggestion only* — human review is mandatory, matching
CLAUDE.md's Absolute Boundaries). Reused all existing architecture, models,
state, tools, and evaluation code rather than rebuilding; this was a rename/
refactor pass, not a new build.

Decisions made and why (kept here so future sessions don't relitigate them):
- **New workflow order**: Goal Agent -> Planner Agent -> Subject Reader Agent
  -> Email Body Reader Agent -> PDF/DOCX Reader Agent -> Medical Extraction
  Agent -> Triage Agent -> Duplicate Search Agent -> Report Generator Agent
  -> Evaluator Agent -> Human Review Agent.
- **Supervisor retained** as a cross-cutting delegation/limits controller
  (`AgentName.SUPERVISOR` still exists) even though it is not one of the 11
  user-named pipeline stages — it enforces delegation depth/attempts/runtime
  per the Context Isolation and Delegation section, which the new workflow
  doesn't replace.
- **Minimum Criteria folded into Medical Extraction** (no separate agent/
  `AgentName` member) — `check_minimum_case_criteria` remains a distinct
  *tool* the Medical Extraction Agent calls; the criteria check itself is
  small enough not to need its own pipeline stage in the new workflow.
- **Missing Information folded into Report Generator** — follow-up-question
  drafting is a tool/output of the Report Generator Agent rather than its
  own pipeline stage, since the new workflow names only 11 agents.
- **`GoldenCaseCategory` enum values kept unchanged** (`COMPLETE`,
  `INCOMPLETE`, `SERIOUS`, `NON_SERIOUS`, etc.) — these are internal
  test/dataset scaffolding labels, not user-facing taxonomy, so renaming
  them would have been pure churn. Only the runtime-facing
  `TriagePriority` (LOW/MEDIUM/HIGH/CRITICAL) and `expected_triage_priority`
  field carry the new taxonomy.
- **New product names**: `DemoInsulex` (type 1 diabetes), `DemoBasalin`
  (type 2 diabetes), `DemoGlutide` (type 2 diabetes) — replacing
  `DemoGluca`/`DemoCardolol`/`DemoZanix`. All fictional, per the Absolute
  Boundaries.
- **New indicator vocabulary**: `TriageIndicator` gained
  `SEVERE_HYPOGLYCEMIA` and `SEVERE_INJECTION_SITE_REACTION` alongside the
  original hospitalization/life-threatening/death/disability/congenital-
  anomaly/other-medically-important set, reflecting injection-specific
  adverse events.
- **Deterministic priority mapping**: `suggest_triage_priority` (new tool,
  `src/tools/domain.py`) maps explicit `TriageIndicator` matches to a
  `TriagePriority` suggestion via documented deterministic rules — never an
  LLM call, never a final determination. The golden-dataset generator
  imports this same function (`_priority_for` helper in
  `scripts/generate_golden_dataset.py`) rather than reimplementing the
  mapping, so golden labels can never drift from the tool's own logic.
- **Test-fixture `AgentName` mapping** (old name -> new name, applied
  wherever old test fixtures referenced a pipeline stage that no longer
  exists as its own agent): `INTAKE` -> `SUBJECT_READER`, `DOCUMENT` ->
  `ATTACHMENT_READER`, `MINIMUM_CRITERIA` -> `MEDICAL_EXTRACTION`,
  `SERIOUSNESS_TRIAGE` -> `TRIAGE`, `GOAL_MANAGER` -> `GOAL_AGENT`.
- Full rewrite of `scripts/generate_golden_dataset.py` (all 21 category
  builders) to the new product/indicator/priority vocabulary; verified
  byte-identical regeneration across two runs with the same seed (only the
  manifest timestamp differs), and re-verified again after this session's
  lint/type fixes.
- `src/tools/quality.py`'s `CompareNarrativeWithFieldsTool` was already
  renamed to `CompareReportWithFieldsTool` in a prior session (Narrative ->
  Report), and `src/models/report.py`/`src/models/triage.py` already carried
  `TriageReport`/`ReportSection`/`TriageFinding`/`TriageResult` — this
  session's work was catching up 6 stale test files that still imported/
  referenced the old names, not renaming the source itself.

Verification actually executed this session (2026-09-19):
```
$ python -m pytest -q
152 passed in 1.90s

$ python -m ruff check .
All checks passed!

$ python -m mypy src config scripts
Success: no issues found in 44 source files

$ python scripts/generate_golden_dataset.py   # 138 cases, 84/27/27 split
$ python scripts/run_golden_evaluation.py     # baseline metrics recomputed, not invented
$ python scripts/seed_synthetic_data.py       # all 7 checks OK
```
Two real (non-cosmetic) mypy bugs were fixed along the way, found only
because mypy was actually run against `src config scripts` (the Makefile's
scope, not bare `mypy .`, which false-positives on `config/settings.py`
being importable under two module names):
- `src/tools/intake.py`: `extract_attachments` decoded `get_payload(decode=True)`
  without checking it was actually `bytes` (email API can return `Message`
  or `Any`); `validate_file`'s `reasons` list had no element-type annotation.
- `src/tools/domain.py`: the ordinal-date branch of `normalize_date`
  reassigned `day`/`month` (already inferred as plain `int` earlier in the
  same function from unpacked regex groups) to `int | None` lookups; renamed
  to `ordinal_day`/`ordinal_month` to avoid the type conflict.
- `src/tools/registry.py`: `ALL_TOOLS`'s dict comprehension over a mixed
  list of `BaseTool[X, Y]` subclasses inferred as `list[object]` without an
  explicit annotation; split into an explicitly-typed
  `_ALL_TOOL_INSTANCES: list[BaseTool[Any, Any]]` first.
- `tests/unit/test_tools_registry.py` and the new `suggest_triage_priority`
  tool: total registered tool count is now **36**, not 35 (the priority
  suggester is a new tool added during the pivot); test updated accordingly.

Still open from this pivot (not yet done):
- `CLAUDE.md` itself still describes the old pharmacovigilance/DemoGluca
  scenario and has not been updated to the diabetes-injection-triage
  business case or the 11-agent workflow.
- `README.md` still needs a rewrite to describe the new business use case,
  workflow, and triage taxonomy.

## Phase 3 — Deterministic Tool Layer — DONE
Built in an earlier session as `src/tools/{base,intake,document,domain,quality,
retrieval,workflow,registry,reference_data}.py`. Re-verified in this session
(2026-09-19) after completing the diabetes-injection-triage pivot (see notes
below): all 36 tools registered (`ALL_TOOLS`), every agent allowlist
references only real tools, ruff and mypy both clean, and every tool has a
passing unit test.
- [x] `BaseTool` harness: typed input/output, timeout, retry, authorization
      check, sanitized error handling, audit event emission
      (`tests/unit/test_tools_base.py`, 6 tests)
- [x] Intake tools: `parse_email`, `extract_attachments`, `validate_file`,
      `validate_file_signature`, `sanitize_filename`, `enforce_file_size`,
      `enforce_page_limit` (`tests/unit/test_tools_intake.py`, 12 tests)
- [x] Document tools: `extract_pdf_text` (real PyMuPDF round-trip),
      `perform_ocr` (honestly reports "unavailable" — no tesseract binary in
      this sandbox, never fabricates OCR text), `assess_ocr_quality`,
      `create_page_citations`, `normalize_source_passages`
      (`tests/unit/test_tools_document.py`, 6 tests)
- [x] Domain tools: `lookup_product_alias`, `lookup_event_term`,
      `normalize_date`, `validate_medical_field`,
      `check_minimum_case_criteria`, `detect_explicit_triage_indicators`,
      `suggest_triage_priority` — the last two implement the LOW/MEDIUM/
      HIGH/CRITICAL AI-suggestion mapping (never a final determination;
      `requires_human_confirmation` always true) (`tests/unit/test_tools_domain.py`,
      16 tests, several cross-checked directly against every matching golden
      dataset case so the tool and its fixtures can't silently drift apart)
- [x] Quality tools: `validate_citations`, `detect_unsupported_claims`,
      `compare_report_with_fields`, `detect_conflicting_values`,
      `validate_goal_completion` (`tests/unit/test_tools_quality.py`, 9 tests)
- [x] Retrieval tool wrappers (naive/placeholder ranking — the real hybrid
      BM25+vector+metadata ranker is Phase 5, not yet built):
      `exact_case_search`, `bm25_search`, `vector_search`, `metadata_filter`,
      `reciprocal_rank_fusion`, `retrieve_case_evidence`
- [x] Workflow tools: `read_case_state`, `write_case_checkpoint`,
      `pause_for_human_review`, `route_case`, `record_audit_event`,
      `create_review_task`
- [x] `src/tools/registry.py`: `ALL_TOOLS` catalog (36 tools) plus
      `authorized_tools_for(agent)` reading per-agent allowlists from
      `config/config.yaml` — Context Isolation requirement (e.g. the
      Duplicate Search agent's allowlist excludes `read_case_state`, so it
      can never see reporter contact details) (`tests/unit/test_tools_registry.py`,
      4 tests)
- [x] pytest **152/152 passed**, ruff clean, mypy clean (44 source files
      under `src config scripts`) — commands and output below

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
