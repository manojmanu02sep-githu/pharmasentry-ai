# PharmaSentry AI — Build Progress

Tracks completed and pending work against the phased plan. Update this file at the
end of every phase. Never mark an item done unless it was actually created/run and
verified (tests executed, not assumed).

Legend: `[x]` done and verified · `[~]` partially done · `[ ]` not started

## Phase 1 — Architecture, Scaffold, Models, State, Demo Case, Initial Tests — DONE
- [x] Repository directory structure (matches CLAUDE.md `Repository` section)
- [x] `progress.md` (this file)
- [x] `README.md`
- [x] `.env.example`
- [x] `requirements.txt`
- [x] `Makefile`
- [x] `docker-compose.yml` (skeleton; app image/full service wiring in Phase 8)
- [x] `config/` — settings loader (`config/settings.py`), `config/config.yaml`
- [x] `src/models/` — typed Pydantic domain models (goal, plan, decisions, evidence,
      extracted fields, minimum criteria, seriousness, duplicates, missing info,
      narrative, evaluation, review, trace/audit events)
- [x] `src/graph/state.py` — typed LangGraph `CaseState` with `operator.add`
      reducers on every append-only list field
- [x] Synthetic demo case (DemoGluca) — email + attachment under `data/synthetic_*`
- [x] `app.py` — placeholder entry point (Streamlit page list only)
- [x] `tests/unit/` — model, state, settings, and demo-data tests (22 tests)
- [x] Ran pytest (22/22 passed), ruff (clean after fixes), mypy (clean, 29 files)
- [x] `.gitignore`, git init, initial commit (`c2aaede`)

Deferred from the literal Phase 1 file list, tracked as later phases instead:
`scripts/seed_synthetic_data.py` and `scripts/run_golden_evaluation.py` exist
as honest placeholders (no invented output) so `make seed` / `make eval` run;
full implementations land in Phase 2 and Phase 6/11. `docs/*` subfolders and
`.claude/*` exist with `.gitkeep` only — populated in Phase 12 / as needed.

## Phase 2 — Synthetic Data & Golden Dataset — DONE
- [x] Synthetic data generator script (`scripts/generate_golden_dataset.py`) —
      deterministic (seeded RNG, `--seed` default 42), fully offline, no LLM
      calls; re-running reproduces byte-identical output (verified: sha256 of
      a sample case file matched across two runs)
- [x] 100 labeled records: 20 complete, 20 incomplete, 15 serious, 15
      non-serious, 10 exact-duplicate + 10 near-duplicate (5 two-member
      families each), 5 conflicting email-vs-attachment, 5 non-safety —
      matches the distribution given for this phase exactly
- [x] Duplicate-family-aware 80/20 train/test split (`family_aware_split` in
      the generator) — verified zero family leakage
      (`validate_no_family_leakage() == []`) and full coverage (every
      case_id appears in exactly one of train/test)
- [x] Golden dataset schemas (`src/evaluation/schemas.py`: `GoldenCase`,
      `GoldenCaseCategory`, `ExpectedMinimumCriteria`, `DatasetSplit`,
      `DatasetManifest`) and read-only loaders (`src/evaluation/golden_loader.py`)
- [x] Baseline metrics framework (`src/evaluation/metrics.py`): classification
      P/R/F1, field-level extraction P/R/F1, seriousness
      sensitivity/specificity, duplicate precision@5/recall@5 (+ MRR as a
      bonus, anticipating Phase 5), citation coverage, unsupported-claim rate
- [x] `scripts/run_golden_evaluation.py` rewritten to actually run the
      metrics framework against the golden test split (20 cases) using a
      clearly-labeled **naive, non-agent baseline**; writes real computed
      numbers to `evaluations/results/phase2_baseline_metrics.json`
      (regenerated each run, gitignored — not a claim about agent quality)
- [x] Tests: `tests/unit/test_evaluation_schemas.py`,
      `tests/unit/test_metrics.py`,
      `tests/integration/test_golden_dataset_integrity.py`,
      `tests/integration/test_generate_golden_dataset.py`,
      `tests/integration/test_run_golden_evaluation.py` (38 new tests)
- [x] Ran pytest (60/60 passed total), ruff (clean), mypy (clean, 35 files
      incl. `scripts/`)
- [x] `pyproject.toml`/Makefile updated (`mypy` now also checks `scripts/`;
      new `make dataset` target)

Deviations/assumptions logged:
- CLAUDE.md's fuller category list also mentions poor-OCR, multilingual, and
  prompt-injection golden cases. This phase's explicit instruction gave a
  concrete 8-category/100-case distribution that does not include those
  three; deferred them to Phase 3 (added alongside the OCR quality checker
  and prompt-injection guardrail they need to be meaningfully labeled and
  scored against) and to the Phase 10 security/red-team suite.
  `expected_extracted_fields` uses plain expected string/None values (not
  runtime `FieldValue` objects with citations) since there is no run to
  cite yet; runtime citation grounding is scored once the agents exist.
- Within this dataset's category taxonomy, "serious" and "incomplete" are
  kept as separate dimensions from each other and from "complete" (each
  category isolates one signal cleanly for metric testing) rather than
  modeling every real-world combination (e.g. a serious case with missing
  dose) — a deliberate simplification of the synthetic distribution, not a
  constraint on future agents.
- Citation coverage / unsupported-claim rate are demonstrated on the golden
  dataset's own `expected_narrative_facts` (grounded by construction, so
  coverage=1.0/unsupported=0.0), since there is no Narrative Agent yet to
  generate real narratives to score (Phase 6).

## Phase 3 — Deterministic Tools & Guardrails
- [ ] Email parser, attachment extractor, file validator, filename sanitizer
- [ ] PDF extractor, OCR adapter + quality checker
- [ ] Citation builder, citation checker, unsupported-claim checker
- [ ] Product/event lookup, date normalizer
- [ ] State reader, review pause, audit logger tools
- [ ] Prompt-injection detection guardrail, output encoding

## Phase 4 — Memory Layers & Isolation
- [ ] Working memory (LangGraph state)
- [ ] Episodic memory (run status, errors, reviewer corrections, versions)
- [ ] Semantic memory (synthetic cases, aliases, vocabulary, retrieval)
- [ ] Procedural memory (prompts, schemas, policies)
- [ ] Case isolation + access logging enforcement tests

## Phase 5 — Hybrid Retrieval & Evaluation
- [ ] BM25 index, vector index (FAISS), metadata filter
- [ ] Deterministic hybrid ranker, top-5 candidates w/ evidence
- [ ] Retrieval metrics: precision@5, recall@5, MRR

## Phase 6 — Agents (one at a time, mock-LLM tested)
- [ ] Goal Manager, Planner, Supervisor
- [ ] Intake, Document, Medical Extraction
- [ ] Minimum Criteria, Seriousness Triage
- [ ] Duplicate Agent, Missing Information Agent
- [ ] Narrative Agent, Evaluator Agent, Human Review Controller

## Phase 7 — Graph Wiring
- [ ] Full LangGraph conditional graph + checkpointing
- [ ] HITL pause/resume
- [ ] Retry-once-then-escalate enforcement

## Phase 8 — Streamlit UI
- [ ] Dashboard, New Case Intake, Case Workspace, Planning/Agent Progress,
      Duplicate Review, Human Review Queue, Evaluation, Observability,
      System Configuration, About/Limitations

## Phase 9 — Observability
- [ ] OTel instrumentation, `/health`, Prometheus metrics, Grafana config
- [ ] In-app trace viewer, audit JSON export, PHI-safe structured logs

## Phase 10 — Full Test Suite
- [ ] Security/red-team suite (injection, tool-arg injection, cross-case access,
      approval bypass, endless-loop, unauthorized external action)
- [ ] Load test script

## Phase 11 — Quality Gate
- [ ] Ruff, mypy, pytest, golden evaluation, smoke test all passing — fix failures

## Phase 12 — Governance & Docs
- [ ] Threat model, governance docs, runbook, limitations, README finalization,
      screenshots

## Notes / Deviations From Spec
(Record any safe-default assumptions here as they are made.)
