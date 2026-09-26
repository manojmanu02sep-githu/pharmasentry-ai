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

## Phase 4 — Memory — DONE
Implemented all four memory tiers required by CLAUDE.md's Memory section as
thin, logged wrappers (`(result, MemoryEvent)` tuples, same convention as
`BaseTool`) rather than a new bespoke framework, so each tier stays
auditable and reuses existing typed models (`MemoryEvent`, `MemoryTier`,
`MemoryOperation` in `src/models/audit.py` / `src/models/enums.py`).
- [x] Working memory (`src/memory/working.py`): read/write over the current
      LangGraph `CaseState` dict passed in by the caller. Case isolation
      enforced locally (`_enforce_case_isolation`, mirroring
      `src/tools/workflow.py`'s pattern) — raises `ToolAuthorizationError`
      on any cross-case read or write.
- [x] Episodic memory (`src/memory/episodic.py`): `EpisodicMemoryStore`,
      SQLite-backed (stdlib `sqlite3`, see deviation below) satisfying the
      same `CaseStateStore` Protocol as `InMemoryCaseStateStore`. Stores
      approved prior run status, an append-only history of status
      changes/errors/reviewer corrections, and supports
      `purge_expired(retention_days)` for the CLAUDE.md-required retention
      control. Cross-case state and history access both blocked. History
      summaries are defensively truncated to 2000 chars so a runaway or
      hostile summary can never be persisted whole (belt-and-suspenders
      alongside the "no chain-of-thought" rule — callers are expected to
      pass concise decision summaries only).
- [x] Semantic memory (`src/memory/semantic.py`): read-only retrieval over
      synthetic reference content that already exists as a single source
      of truth — product aliases/canonical names and event vocabulary from
      `src/tools/reference_data.py`, and the full synthetic case corpus via
      `src.evaluation.golden_loader.load_all_cases()`. Never forks or
      duplicates that data; never writes. This is global reference content
      rather than case state, so there is no case-isolation check here
      (documented in-module) — access is still logged via `MemoryEvent`.
      Which fields an agent may see (e.g. never reporter contact details
      for the Duplicate agent) is enforced downstream in
      `src.retrieval.corpus`, not here.
- [x] Procedural memory (`src/memory/procedural.py`): read-only accessors
      for prompts/schema versions, the LangGraph node order, per-agent tool
      allowlists, delegation/runtime/token limits, the retrieval weight
      policy, and the retention policy — all sourced from the single
      `config/config.yaml`, never hard-coded a second time.
- [x] pytest: `test_memory_working.py` (4), `test_memory_episodic.py` (9),
      `test_memory_semantic.py` (3), `test_memory_procedural.py` (6) — 22
      tests, all passing (see combined Phase 4+5 run below).

Deviation from the CLAUDE.md stack list, documented here as instructed:
episodic memory uses stdlib `sqlite3` directly rather than SQLAlchemy. The
CLAUDE.md stack lists "PostgreSQL with SQLite local fallback" for the
application's persistence generally; for this narrow append-only
key/history store, a raw `sqlite3` connection satisfies that same SQLite
requirement with no added dependency, and the store is already isolated
behind the `CaseStateStore` Protocol so a SQLAlchemy/Postgres-backed
implementation can be swapped in later without touching callers.

## Phase 5 — Hybrid RAG — DONE
Implemented the full BM25 + vector + metadata retrieval stack required by
CLAUDE.md's Hybrid RAG section, on top of (not duplicating) the Phase 3
placeholder retrieval tools in `src/tools/retrieval.py`.
- [x] `src/retrieval/embeddings.py`: an `EmbeddingProvider` interface with
      a deterministic default (hash-based, reproducible, L2-normalized —
      no ML dependency required to run or test) and a lazy-imported
      `SentenceTransformerEmbeddingProvider` for real embeddings, selected
      via `config/settings.py`'s new `EmbeddingProviderName` enum /
      `embedding_provider` setting. `sentence-transformers`/`torch` are
      intentionally not installed in this environment — the provider is
      only imported if actually selected.
- [x] `src/retrieval/corpus.py`: builds the retrieval corpus from the
      golden dataset only (CLAUDE.md: "the retrieval corpus must contain
      synthetic cases only"), combining email + attachment text per case
      and explicitly excluding `reporter.name` from indexed text.
- [x] `src/retrieval/bm25_index.py`: `PersistentBm25Index` — corpus
      persisted as JSON (see deviation below), `BM25Okapi` ranking rebuilt
      in memory on load/search.
- [x] `src/retrieval/faiss_index.py`: `PersistentFaissIndex` over
      `faiss.IndexFlatIP`, native FAISS binary serialization, with
      `FaissIndexMismatchError` raised if a saved index's embedding
      provider doesn't match the one loading it. `faiss-cpu` installed
      into `.venv`; the corresponding test module opens with
      `pytest.importorskip("faiss")` so the suite still runs (skipping
      only that file) in an environment without it.
- [x] `src/retrieval/hybrid_ranker.py`: `weighted_reciprocal_rank_fusion` —
      the one documented, deterministic combination method CLAUDE.md
      requires ("combine rankings using a documented deterministic
      method"), generalizing the existing unweighted RRF tool
      (`src.tools.retrieval.ReciprocalRankFusionTool`) to per-ranking
      weights; with all weights at 1.0 it reduces to exactly that tool's
      formula. `HybridRanker.from_config` reads the real
      `bm25_weight`/`vector_weight`/`top_k` values from
      `config/config.yaml` (0.5 / 0.5 / 5) via procedural memory, rather
      than a second, separately-tuned combiner.
- [x] `src/retrieval/duplicate_search.py`: `DuplicateSearchService` —
      returns up to top-5 `DuplicateCandidate` objects (matching fields,
      conflicting fields, BM25/vector/combined scores, evidence snippets)
      and a `RetrievalEvent`. Never merges cases and exposes no field that
      could constitute an auto-merge or final decision — enforced by test
      (`test_search_returns_candidates_never_merges` asserts no `merged`
      attribute exists on the result).
- [x] `src/retrieval/retrieval_evaluation.py` +
      `scripts/run_retrieval_evaluation.py`: real precision@5, recall@5,
      and MRR for BM25-only, vector-only, and hybrid retrieval, computed
      against the full 138-case golden dataset (queries restricted to the
      cases that actually have a non-empty `expected_duplicate_matches`,
      per the "don't modify the dataset" constraint — this only reads it
      via the existing `golden_loader`). RAG never makes the final
      duplicate decision — it only ever returns ranked candidates for the
      Duplicate agent / human reviewer.
- [x] `scripts/build_retrieval_index.py`: builds and persists the BM25 and
      FAISS indexes from the golden corpus to `evaluations/results/` (or a
      given path), for reuse outside of tests.
- [x] pytest: `test_retrieval_corpus.py` (3), `test_retrieval_bm25_index.py`
      (3), `test_retrieval_faiss_index.py` (4),
      `test_retrieval_hybrid_ranker.py` (4),
      `test_retrieval_duplicate_search.py` (2),
      `test_retrieval_evaluation.py` (3), `test_retrieval_embeddings.py`
      (5) — 24 tests, all passing.
- [x] Retrieval evaluation actually executed
      (`.venv/bin/python scripts/run_retrieval_evaluation.py`, output
      written to `evaluations/results/phase5_retrieval_metrics.json`):
      16 queries evaluated (every golden-dataset case with a non-empty
      `expected_duplicate_matches`); BM25, vector, and hybrid each scored
      precision@5 = 0.2, recall@5 = 1.0, MRR = 1.0. All three methods
      land on identical numbers on this dataset because with exactly one
      expected duplicate per query, every method places it at rank 1 —
      precision@5 is mechanically capped at 1/5 whenever only one relevant
      document exists per query. This is a real, executed measurement,
      not an invented one; it is not evidence that BM25/vector/hybrid are
      equivalent in general, only on this corpus's current duplicate
      families.

Deviation from the CLAUDE.md stack list, documented here as instructed:
`PersistentBm25Index` persists its corpus as JSON rather than a pickled
`BM25Okapi` object. `rank-bm25`'s index has no native serialization; a
pickle round-trip would work but means deserializing untrusted or
corrupted state could execute arbitrary code on load, which conflicts with
this project's security posture (CLAUDE.md: "treat uploaded content as
untrusted data"; file validation and path sanitization are required
elsewhere). Persisting the plain-text corpus as JSON and rebuilding the
`BM25Okapi` ranking in memory on load avoids that risk entirely and keeps
the on-disk index human-auditable, at the cost of a small rebuild step on
load — an acceptable tradeoff at this corpus size (138 synthetic cases).

While running `mypy` for Phase 4/5 (see combined run below), one
pre-existing, unrelated issue surfaced and was fixed: `config/` had no
`__init__.py`, which caused mypy to resolve `config/settings.py` as two
different module names ("settings" and "config.settings") and abort before
checking anything — added `config/__init__.py` (mirroring the existing
`src/__init__.py`), matching the `packages = ["src", "config", "scripts"]`
setting already in `pyproject.toml`. Two genuine type errors in the new
`src/memory/semantic.py` were also found and fixed: a dict literal that
mypy widened to a supertype not matching the declared return annotation
(fixed with an explicit variable annotation), and a `tuple[GoldenCase, ...]`
returned where `list[GoldenCase]` was declared (fixed by correcting the
return annotation to match `load_all_cases()`'s actual tuple return type).

Verification, run from the repository root:
```
.venv/bin/pytest -q
# 198 passed in 4.79s   (152 from Phases 1-3 + 46 new: 22 memory + 24 retrieval)

.venv/bin/ruff check .
# All checks passed!

.venv/bin/mypy src config scripts
# Success: no issues found in 58 source files
```

## Phase 6 — LLM Provider Abstraction — [~] (mock only)
- [x] `src/llm/base.py` — `LLMProviderProtocol` with a generic
      `complete(prompt, response_model: type[T], *, context=None) ->
      tuple[T, LLMCallMetadata]`, Pydantic structured output throughout.
- [x] `src/llm/mock.py` — deterministic mock provider (no network calls,
      reproducible output for tests/CI).
- [x] `src/llm/factory.py` — `get_llm_provider(settings)`; mock is
      selectable and works end-to-end (proven by the Phase 8 e2e test).
- [ ] Real cloud provider (Anthropic) — **not implemented**.
      `get_llm_provider` raises `LLMUnavailableError` for any non-mock
      `settings.llm_provider` value rather than fabricating a client; this
      is an honest not-yet-built gap, not a silent fallback.
- [ ] No dedicated `tests/unit/test_llm_*` file yet — the mock provider is
      only exercised indirectly via the agents that call it.

## Phase 7 — Agents — [~] (thin slice, one integration test only)
- [x] All 12 agents implemented as `run(state: dict) -> dict` partial-state
      nodes in `src/agents/`: `goal_agent`, `planner`, `supervisor`,
      `subject_reader`, `email_body_reader`, `pdf_docx_reader`,
      `medical_extraction`, `triage`, `duplicate_search`,
      `report_generator`, `evaluator`, `human_review`. Shared LLM-call/audit
      helpers factored into `src/agents/_common.py`.
- [x] Each agent returns structured decision/evidence/confidence/
      decision_summary/next_action/requires_human_review fields per
      CLAUDE.md's Reasoning requirement (verified by code review and by the
      e2e test's assertions on citations, confidence, and evaluation
      output — not by a full per-agent unit-test suite).
- [x] `human_review.run()` + `apply_review_decision()` implement the
      pause/approve/reject/request-changes gate; the graph never
      auto-resumes past human review (hard edge to `END` — see Phase 8).
- [ ] **No per-agent unit tests exist yet** (`tests/unit/` has no
      `test_agents_*` files). Coverage today is one integration test
      (`tests/integration/test_demo_case_e2e.py`) that runs the whole
      12-node pipeline once, on the one synthetic demo case. This is a real
      gap: individual agent edge cases (LLM timeout, malformed evidence,
      conflicting fields) are not yet covered by dedicated tests.

## Phase 8 — LangGraph Orchestration — [~] (demo case only, one e2e test)
- [x] `src/graph/state.py` — typed `CaseState`; `src/graph/runner.py` —
      `build_graph()`/`compiled_app()`/`run_case()` wiring the 12 nodes with
      `MemorySaver` checkpointing (in-memory, process-local — same posture
      as `InMemoryCaseStateStore`); `src/graph/routing.py` — conditional
      routing after `subject_reader`/`email_body_reader`/`pdf_docx_reader`
      (unsupported/unsafe file -> human review); `src/graph/demo_case.py` —
      `build_demo_case_initial_state()` for CLAUDE.md's synthetic demo case.
- [x] `ruff check .` and `mypy src config scripts` clean (0 errors) across
      all of the above, including a first-ever project-wide mypy run this
      session (fixed 7 real type errors: a non-generic `call_llm_or_none`
      losing type info, two `Any | None`/`FieldValue | None` narrowing
      issues in `medical_extraction.py`, and one documented
      `# type: ignore[call-overload]` for a genuine LangGraph-stub vs.
      intentional plain-`dict`-node-signature mismatch in `runner.py`).
- [x] `tests/integration/test_demo_case_e2e.py` (new) — runs
      `build_demo_case_initial_state()` through `run_case()` and asserts:
      `goal_status == AWAITING_HUMAN`, `current_step == "human_review"`,
      zero errors, non-empty evidence passages, grounded citations on
      `product.product_name` and `outcome.hospitalized`,
      `meets_minimum_criteria is True`, a real `EvaluationResult` with
      `passed is True` and `evidence_coverage == 1.0`, and
      `review_status.decision == "pending"`. Verified:
      `pytest tests/integration/test_demo_case_e2e.py -q` → **1 passed**.
- [x] Full suite after all of the above: `pytest -q` → **199 passed**
      (198 pre-existing + this new test), `ruff check .` → all checks
      passed, `mypy src config scripts` → no issues in 81 source files.
- [ ] Only the one demo case has been run through the graph. Conditional
      routes for unsupported files / low-OCR / non-safety / missing-criteria
      / duplicate / retry-then-escalate exist in `routing.py` but are not
      yet each covered by their own integration test.

## Phase 9 — Evaluation Framework — [ ]
## Phase 10 — Observability and Traceability — [ ]
## Phase 11 — Security, Guardrails, and Red Teaming — [ ]
## Phase 12 — Streamlit User Interface — [~] (Case Workspace only, server-boot verified — not interactively click-tested)
- [x] `app.py` wired to a real sidebar page nav (`st.sidebar.radio`); the
      other 9 pages (Dashboard, New Case Intake, Planning and Agent
      Progress, Duplicate Review, Human Review Queue, Evaluation,
      Observability and Traceability, System Configuration, About and
      Limitations) are explicit `st.info(...)` placeholders, not
      implemented.
- [x] `src/ui/case_workspace.py` (new) — "Run demo case" button calling
      `build_demo_case_initial_state()` + `run_case()`, state held in
      `st.session_state`; renders original evidence beside extracted
      fields (value/confidence/citation-count/conflict-status table per
      CLAUDE.md's Case Workspace requirement), minimum-criteria status,
      AI-suggested triage findings, duplicate candidates, missing-info
      follow-up draft, cited narrative, evaluation results, and
      Approve/Reject/Request-changes controls gated on
      `goal_status == AWAITING_HUMAN`. A decision calls
      `apply_review_decision()` directly (never re-invokes the graph, since
      `human_review` has a hard edge to `END`) and reruns to show the
      updated `review_status`.
- [x] App boot verified twice via background `streamlit run app.py`
      (project's own `.venv` lacks Streamlit; used
      `/home/labuser/venv/bin/python3 -m streamlit run app.py
      --server.headless true`) — server started cleanly (Uvicorn started,
      static shell served over `curl`), no import/render traceback.
- [ ] **Not done**: an actual interactive browser click-through (clicking
      "Run demo case", visually checking the rendered tables/citations,
      clicking Approve/Reject/Request-changes and confirming the UI
      updates). Only server-boot-level verification was performed. This
      was explicitly deprioritized under a user time constraint in favor of
      the real, executed integration test above, and is flagged here
      rather than claimed as done.
## Phase 13 — API and Service Layer — [ ]
## Phase 14 — CI/CD and Continuous Evaluation — [ ]
## Phase 15 — Documentation — [ ]

## Session note (2026-09-26): lint/type cleanup + Case Workspace + e2e test
This session (continuing on top of the already-existing but previously
undocumented Phase 6/7/8 thin slice): fixed all remaining `ruff` `E501`/
`I001` violations across `subject_reader.py`, `supervisor.py`, `triage.py`,
`routing.py`, `runner.py` (73 -> 0 project-wide); ran `mypy` for the first
time in this project's history and fixed all 7 real errors it found (see
Phase 8 above) rather than suppressing them, with one narrowly-scoped,
commented `# type: ignore[call-overload]` for a genuine stub/runtime-design
mismatch; built and wired the Case Workspace UI page; wrote and ran the
first pipeline-level integration test. Also confirmed (separately, earlier
in this session) that the previously-planned RAG retrieval refactor —
threading the configured `EmbeddingProvider` through `VectorSearchTool`,
`DuplicateSearchService.from_settings()`, and `run_retrieval_evaluation()`,
plus PHI-safe structured logging in the retrieval modules — was already
fully implemented in the working tree (uncommitted); re-verified it still
passes lint/type/test checks alongside everything else. Explicitly not
done: real interactive browser verification of the new UI, per-agent unit
tests, and Phases 9-11/13-15.
## Final Verification — [ ]

(Each phase above gets its own detailed section, filled in as completed —
see below. Sections are appended in order; nothing is marked `[x]` above
until its detailed section below has actually been executed and verified.)
