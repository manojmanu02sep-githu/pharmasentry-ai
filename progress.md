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

## Phase 2 — Synthetic Data & Golden Dataset
- [ ] Synthetic data generator script
- [ ] 100+ labeled records (complete, incomplete, serious, non-safety, exact/near
      duplicate, conflicting, poor OCR, multilingual, prompt-injection)
- [ ] Duplicate-family-aware split (no leakage)
- [ ] Golden dataset schema + storage under `evaluations/golden_dataset/`

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
