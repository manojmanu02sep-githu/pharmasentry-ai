# PharmaSentry AI (Educational Prototype)

> **This is an educational prototype. It uses only synthetic data. It is
> NOT a medical device, safety database, regulatory submission system, or
> clinical decision system.** No output from this system is a final
> medical, causality, seriousness, expectedness, validity, reportability,
> or regulatory decision. Every consequential output requires human review
> and approval. See `docs/threat-model/` and "Limitations" below.

## What this is

PharmaSentry AI is a bounded, goal-driven multi-agent prototype that
processes synthetic adverse-event emails and attachments and prepares a
**preliminary** safety-case package: parsed evidence, structured fields
with citations, minimum-case-criteria status, explicit seriousness
indicators, missing/conflicting information, potential duplicate
candidates, a source-cited narrative draft, an editable follow-up draft,
a quality evaluation, and a complete audit trace — then **pauses for
mandatory human review**. It never merges cases, sends email, updates a
production system, or submits anything automatically.

Build status is tracked phase-by-phase in [`progress.md`](progress.md).
This README documents what exists **today**; do not read planned
capabilities as already implemented.

## Architecture

LangGraph orchestrates 13 bounded agents over one typed, checkpointed
`CaseState` (see `src/graph/state.py`):

```
START -> Goal Manager -> Planner -> Supervisor -> Intake -> Document ->
Extraction -> Minimum Criteria -> Seriousness -> Duplicate Search ->
Missing Information -> Narrative -> Evaluator -> Human Review -> END
```

Conditional routes (implemented as the graph is built out — see
`progress.md`): unsupported/unsafe file -> reject safely; low OCR quality
-> manual document review; non-safety content -> human closure
confirmation; missing minimum criteria -> follow-up + incomplete-case
queue; potential duplicate -> duplicate review queue; validation failure
-> retry once then manual review; LLM unavailable -> deterministic
fallback or manual review.

### The eight required agentic capabilities

| # | Capability | Where |
|---|---|---|
| 1 | Goal | `src/models/goal.py` (`CaseGoal`, `SuccessCriterion`) |
| 2 | Planning | `src/models/plan.py`, Planner Agent (Phase 7) |
| 3 | Reasoning | `src/models/reasoning.py` (`AgentDecision` — decision, evidence, confidence, decision_summary, next_action, requires_human_review; no chain-of-thought is ever persisted) |
| 4 | LLM | `src/services/llm/` provider abstraction: configured cloud provider + deterministic mock (Phase 6) |
| 5 | Memory | `src/memory/{working,episodic,semantic,procedural}.py` (Phase 4) |
| 6 | Tools | `src/tools/` — typed, timeout-bound, audited (Phase 3) |
| 7 | Evaluation | `src/evaluation/` — Evaluator Agent, retry-once-then-escalate (Phase 6/11) |
| 8 | Observability/Traceability | `src/observability/`, `src/models/audit.py` (Phase 9) |

### Repository structure

```text
pharmasentry-ai/
  CLAUDE.md               specification this build follows
  README.md
  progress.md             phase-by-phase build status (source of truth)
  .env.example
  docker-compose.yml
  app.py                   Streamlit entry point (placeholder pages, Phase 1)
  src/
    agents/                agent implementations (Phase 6)
    graph/                 LangGraph state + graph (state.py done, graph.py Phase 7)
    models/                typed Pydantic domain models (done, Phase 1)
    tools/                 deterministic tools (Phase 3)
    memory/                working/episodic/semantic/procedural (Phase 4)
    retrieval/             hybrid BM25 + vector retrieval (Phase 5)
    evaluation/             evaluator agent + metrics (Phase 6/11)
    guardrails/            validation, sanitization, injection detection (Phase 3)
    observability/         tracing, metrics, trace viewer (Phase 9)
    services/               LLM provider abstraction, DB access (Phase 4/6)
    ui/                     Streamlit pages (Phase 8)
  data/
    synthetic_emails/        synthetic demo case email
    synthetic_attachments/   synthetic demo case attachment
    synthetic_cases/         synthetic retrieval corpus (Phase 2)
  tests/{unit,integration,e2e,security,evaluation}/
  evaluations/{golden_dataset,results}/
  config/                   settings.py (env config), config.yaml (policy)
  scripts/                  seed / evaluation entry points
  docs/{architecture,governance,threat-model,runbooks}/
  .claude/{agents,skills,hooks,rules}/
  .github/workflows/
```

## Setup

Requires Python 3.11+ (developed/tested on 3.14).

```bash
make setup      # creates .venv, installs requirements.txt, copies .env.example -> .env
make seed       # verifies/prepares synthetic demo data
make test       # runs pytest
make lint       # ruff
make typecheck  # mypy
make eval       # golden evaluation (placeholder until Phase 2/11)
make run        # streamlit run app.py (placeholder pages until Phase 8)
make docker-up  # docker compose up --build
```

`LLM_PROVIDER=mock` in `.env.example` is the default — the app and its
tests run fully offline with deterministic outputs. Set
`LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` only if you intend to
exercise the real provider once it is wired up (Phase 6).

## Current status (Phase 1)

Implemented and tested:
- Full repository scaffold matching the structure above.
- Typed Pydantic domain models covering goal, plan, reasoning, evidence,
  intake, extraction, minimum criteria, seriousness, duplicates, missing
  information, narrative, evaluation, review, and audit/observability
  events (`src/models/`).
- Typed LangGraph `CaseState` (`src/graph/state.py`) with an
  `create_initial_state` factory.
- Synthetic demo case (fictional product **DemoGluca**, fictional people
  and organizations) under `data/synthetic_emails/` and
  `data/synthetic_attachments/`.
- Configuration: `.env.example`, `config/settings.py`
  (`pydantic-settings`), `config/config.yaml` (agent tool allowlists,
  graph node order, limits).
- Initial unit tests for the above (`tests/unit/`) — see **Tests actually
  executed** below.

Not yet implemented (see `progress.md` for the full phase plan): agents,
LangGraph graph wiring, tools, memory layers, hybrid retrieval, Streamlit
UI, observability instrumentation, golden dataset, security/red-team
suite. `app.py` currently renders a placeholder page list only.

## Tests actually executed

Run in this environment via `pytest -v` inside `.venv` on 2026-09-19.
Exact command and result are reported at the end of this build turn — see
the assistant's final summary for that run's pass/fail count. Do not trust
a stale count here if you have re-run the suite yourself; run `make test`.

## Evaluation metrics

None yet — the golden dataset (Phase 2) and Evaluator Agent (Phase 6) do
not exist yet, so no classification/retrieval/quality metrics can be
honestly reported. `scripts/run_golden_evaluation.py` currently exits
without inventing numbers, in line with the "never invented" requirement.

## Security controls

Implemented: `.env`-based secrets (never committed, see `.gitignore`),
`.env.example` with no real credentials, synthetic-only data policy
documented in `data/SYNTHETIC_DATA_NOTICE.md`. File validation, filename
sanitization, prompt-injection detection, RBAC-ready authorization, and the
rest of the "Security and Governance" checklist in `CLAUDE.md` are
implemented in Phase 3 and audited in `docs/threat-model/` (Phase 12).

## Limitations

This build phase provides scaffolding and typed contracts only — there is
no running end-to-end case pipeline yet, no UI logic, and no LLM calls.
Treat every field and enum here as the agreed data contract for the phases
that follow, not as a working product. See `progress.md` for exactly
what remains.

## Troubleshooting

- **`ModuleNotFoundError` running tests directly with `python -m pytest`**:
  use `make test` (uses `.venv`) or ensure `pyproject.toml`'s
  `pythonpath = ["."]` is picked up (requires running pytest from the repo
  root).
- **`streamlit` not installed**: `make setup` installs it from
  `requirements.txt`; the UI is a placeholder until Phase 8 regardless.
- **Postgres**: not required for local development — `DB_BACKEND=sqlite`
  is the default fallback and needs no Docker service.
