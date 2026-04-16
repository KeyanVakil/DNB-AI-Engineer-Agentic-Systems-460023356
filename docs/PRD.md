# FinSight Agents — Product Requirements Document

A reference multi-agent application for financial insight generation, built to
demonstrate the engineering patterns DNB AI Tech is scaling across the group:
LangGraph-orchestrated agents, MCP tool integration, code/LLM/human evaluations,
MLflow experiment tracking, OpenTelemetry observability, and drift detection — all
runnable locally with `docker compose up`.

---

## 1. Project Overview

### What it does

**FinSight Agents** is a multi-agent system that produces personalized financial
insight reports for retail banking customers. Given a customer ID, a coordinator
agent plans the work, dispatches specialist agents (transaction analyst, portfolio
analyst, market researcher, compliance reviewer), and a writer agent synthesizes
their findings into a structured report rendered in a web UI.

The system ships with:

- A **LangGraph** orchestration graph with five specialist agents and one writer
- Three **MCP servers** (customer-data, market-data, compliance-rules) that expose
  enterprise-style data through the Model Context Protocol
- An **evaluation harness** with code-based assertions, LLM-as-judge scoring, and
  a human review queue, all tracked in **MLflow**
- **OpenTelemetry** traces, metrics, and logs covering every agent step, tool
  call, and token, exported to a local Jaeger + Prometheus + Grafana stack
- **Drift detection** comparing current run distributions (latency, cost, judge
  scores, tool-call patterns) against rolling baselines, with alerts surfaced in
  Grafana
- A **Streamlit** UI for triggering runs, browsing reports, inspecting agent
  traces, and reviewing evaluations

### Why it's relevant to DNB

The job description says AI Tech's mission is to "scale agentic AI across the
group" and contribute "shared patterns, templates, and reference repos." This
project is built as exactly that: a self-contained reference template that
demonstrates production-grade patterns for the listed responsibilities — design
multi-agent applications, integrate with enterprise systems via MCP, build
evaluations, operationalize with observability and drift detection.

The financial domain is intentional. Banking insight generation involves the
exact tradeoffs the job mentions: cost vs. latency vs. capability across model
choices, sensitive data handled through tool layers (not pasted into prompts),
and compliance review baked into the workflow rather than bolted on.

### The problem it solves

Real teams adopting agentic systems hit three repeated failure modes:

1. **Demos don't survive contact with production** because there's no eval
   harness to detect quality regressions when you change a prompt or model
2. **Agent runs are opaque** — when an answer is wrong, nobody can tell which
   agent made the bad decision or which tool returned bad data
3. **Drift is invisible** — model providers update silently, tool schemas
   change, and quality degrades over weeks before anyone notices

FinSight Agents is the worked example of how to address all three from day one.

---

## 2. Technical Architecture

### System architecture

```
                         ┌──────────────────────────┐
                         │   Streamlit Web UI       │
                         │   (run, view, review)    │
                         └────────────┬─────────────┘
                                      │ HTTP
                         ┌────────────▼─────────────┐
                         │   FastAPI Orchestrator   │
                         │   (run lifecycle, eval)  │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │   LangGraph State Graph  │
                         │                          │
                         │   Coordinator            │
                         │   ├─► Transaction Agent  │
                         │   ├─► Portfolio Agent    │
                         │   ├─► Market Researcher  │
                         │   ├─► Compliance Agent   │
                         │   └─► Report Writer      │
                         └──┬──────┬────────┬───────┘
                            │      │        │  (MCP over stdio/HTTP)
              ┌─────────────▼┐  ┌──▼──────┐ ┌▼─────────────┐
              │ Customer MCP │  │Market   │ │ Compliance   │
              │ (Postgres)   │  │MCP      │ │ MCP (rules)  │
              └──────────────┘  └─────────┘ └──────────────┘

              LLM:  Ollama (local) | optional OpenAI/Anthropic/Bedrock
              Memory: Postgres (run state, conversation, embeddings via pgvector)
              Eval:   MLflow tracking server (sqlite + artifacts volume)
              Obs:    OTel Collector → Jaeger (traces), Prometheus + Grafana
```

### Key components

| Component | Responsibility |
|-----------|----------------|
| **Streamlit UI** | Trigger runs, browse reports, view trace timelines, review eval samples |
| **FastAPI service** | REST API: start runs, poll status, fetch reports, submit human reviews, list eval results |
| **LangGraph graph** | Stateful multi-agent orchestration with checkpointing into Postgres |
| **Specialist agents** | Each is a LangGraph node with its own prompt, tool list, and structured output schema |
| **MCP servers** | Tool layer — agents talk to data via MCP only, never SQL or HTTP directly |
| **Evaluator** | Runs code asserts, LLM-judge scoring, drift checks; logs everything to MLflow |
| **OTel collector** | Receives spans/metrics/logs from app and agents; routes to Jaeger and Prometheus |
| **Drift monitor** | Background job comparing current-window distributions to baselines |

### Data flow (single insight run)

1. User picks a customer in the Streamlit UI and clicks **Generate report**
2. UI POSTs to `/runs` on the FastAPI service; service inserts a `run` row and
   kicks off the LangGraph graph in a worker task
3. **Coordinator** node reads the customer profile via MCP, decides which
   specialists to invoke (always all four in the demo, but the planning step is
   real and traced)
4. Specialist nodes execute in parallel where possible; each calls MCP tools,
   returns a structured Pydantic model into graph state
5. **Compliance** node reviews specialist outputs against rules MCP; can request
   revisions (loop back) or approve
6. **Writer** node assembles the final markdown report and structured JSON
7. Run completes; evaluator runs immediately: code asserts against the JSON,
   LLM-judge scores the markdown report, sample is enqueued for human review
8. All of the above emits OTel spans (agent.coordinator, agent.transaction,
   tool.mcp.customer.get_transactions, llm.completion, eval.judge, …) and metrics
   (tokens, latency, cost, judge score)
9. UI polls `/runs/{id}`, then renders the report, the trace tree, and eval
   results

---

## 3. Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Language | **Python 3.12** | Job lists Python first; ecosystem fit for LLM/agent work |
| Multi-agent orchestration | **LangGraph** | Job names LangGraph explicitly; stateful graph with checkpointing fits the workflow |
| Tool layer | **Model Context Protocol (MCP)** | Job explicitly calls out MCP for enterprise integration; we ship three real MCP servers |
| LLM (default, local) | **Ollama** running `llama3.1:8b` | Lets `docker compose up` work with zero API keys; the job doesn't require a cloud LLM |
| LLM (optional) | OpenAI / Anthropic / **AWS Bedrock-compatible** endpoints via LiteLLM | Job mentions Bedrock and multi-cloud; provider is a config switch, not a code change |
| Web UI | **Streamlit** | Job is backend/AI-engineer focused; Streamlit gives a hireable-quality UI without a frontend stack |
| API | **FastAPI** + **Pydantic v2** | Async-first, type-safe, plays well with LangGraph and OTel |
| Memory / state | **Postgres 16** + `pgvector` | LangGraph checkpointer + conversation memory + embedding store in one place |
| Experiment tracking | **MLflow** | Job lists MLflow; we use it for eval runs, prompt versions, and model/config comparisons |
| Observability | **OpenTelemetry** (SDK + Collector) → **Jaeger** + **Prometheus** + **Grafana** | Job lists OTel; the OSS stack runs in docker compose and demonstrates the full pipeline |
| Drift detection | Custom job using `scipy.stats` (KS test, PSI) on metrics from Prometheus | Job calls out drift detection explicitly |
| Container runtime | **Docker** + **docker compose** | Job lists Docker; compose makes the entire stack one-command |
| CI | **GitHub Actions** | Job lists GitHub Actions; we ship lint, type-check, unit, integration, and eval-smoke jobs |
| Code quality | `ruff`, `mypy --strict`, `pytest`, `pytest-asyncio`, `pytest-cov` | Production-grade defaults |

**Deliberately NOT included** (out of scope per the simplicity principle): AWS
AgentCore deployment scripts, Terraform/IaC, Kubernetes manifests, Java/C#
services. The job mentions these as part of DNB's broader stack but a single
demo project shouldn't drag them in. The architecture is designed so the LLM
provider and MCP transports could be swapped to AgentCore/Bedrock with config
changes only.

---

## 4. Features & Acceptance Criteria

### F1. Multi-agent insight generation

A user can trigger an end-to-end insight run for a customer and get back a
structured report.

**Acceptance criteria**
- POST `/runs` with `{customer_id}` returns a `run_id` in <200ms (work happens async)
- The LangGraph graph contains nodes: `coordinator`, `transactions`, `portfolio`,
  `market`, `compliance`, `writer`, with the compliance→specialist revision edge
- Specialists run concurrently where the graph allows (verified in trace)
- The final report contains: spending summary, portfolio summary, market
  context, compliance notes, three actionable recommendations
- Run state is checkpointed to Postgres; killing the worker mid-run and
  restarting resumes from the last completed node

### F2. MCP tool integration

Agents access data exclusively through MCP servers — never direct DB or HTTP.

**Acceptance criteria**
- Three MCP servers are running as separate processes in compose:
  `customer-mcp` (transactions, accounts, profile), `market-mcp` (prices,
  indices, news), `compliance-mcp` (rule lookups, PII checks)
- Each server exposes ≥3 tools with documented schemas
- Agents discover tools via MCP `list_tools`; tool names are not hardcoded in
  agent code
- Replacing an MCP server with a stub at test time requires zero changes to
  agent code (verified by test)
- A failing tool call surfaces as a typed error in graph state, not an exception
  that crashes the run

### F3. Evaluation harness

Every run is evaluated automatically; humans can override.

**Acceptance criteria**
- Code evals: report JSON conforms to schema, recommendation count is in [2,5],
  no PII strings appear in the markdown
- LLM-judge eval: a separate LLM call scores the report on
  {factuality, helpfulness, tone} on 1–5 scales with rationale
- Human review: the UI exposes a queue; reviewers submit `{score, notes,
  approved}` per sample
- All three eval types are logged as runs in MLflow under one experiment per
  prompt version
- A CLI `finsight eval --baseline v1 --candidate v2` prints a regression report

### F4. Cost / latency / capability comparison

Operators can compare model configurations on a fixed eval set.

**Acceptance criteria**
- A `configs/models.yaml` file declares model variants (e.g. `llama3.1:8b`,
  `llama3.1:70b`, `gpt-4o-mini`, `claude-haiku`) with per-token cost
- `finsight bench --config configs/models.yaml --dataset evals/golden.jsonl`
  runs the eval set against each variant
- Per-variant metrics (mean latency, total cost, mean judge score, code-eval
  pass rate) are logged to MLflow and rendered as a comparison table in the UI
- The bench is reproducible: same dataset + same config seed → same MLflow
  metrics within tolerance

### F5. Observability

Every meaningful operation is traceable, measurable, and queryable.

**Acceptance criteria**
- OTel spans cover: HTTP request, each LangGraph node, each MCP tool call,
  each LLM completion (with prompt hash + token counts as attributes)
- Spans are visible in Jaeger at `localhost:16686` within seconds of a run
  finishing
- Prometheus scrapes app metrics: `agent_step_duration_seconds`,
  `llm_tokens_total{kind=in|out,model=…}`, `llm_cost_usd_total`,
  `tool_call_duration_seconds`, `eval_judge_score`
- A pre-built Grafana dashboard shows: runs/min, p50/p95 latency by agent,
  token cost over time, judge-score moving average

### F6. Drift detection

Quality and behavior drift trigger alerts.

**Acceptance criteria**
- A drift job runs every 5 minutes (compose service: `drift-monitor`)
- Compares the last hour of judge scores and tool-call distributions to a
  rolling 24-hour baseline using KS test + PSI
- Drift events are written to the `drift_events` table and to a Prometheus
  gauge; Grafana alert rule fires when `psi > 0.2` or `ks_p < 0.01`
- The UI surfaces a "Drift" badge on affected metrics

---

## 5. Data Models

All persistent data lives in one Postgres database (`finsight`). LangGraph's
checkpointer uses its own schema (`langgraph`); domain tables use `public`.

### Customer-side (seeded fixture data, served by `customer-mcp`)

```
customers (id PK, name, segment, joined_at)
accounts  (id PK, customer_id FK, kind ENUM[checking,savings,credit],
           balance_nok NUMERIC(14,2), opened_at)
transactions (id PK, account_id FK, ts TIMESTAMPTZ, amount_nok NUMERIC,
              merchant TEXT, category TEXT, is_recurring BOOL)
holdings  (id PK, customer_id FK, ticker TEXT, quantity NUMERIC,
           avg_cost_nok NUMERIC)
```

### Market-side (seeded, served by `market-mcp`)

```
prices    (ticker TEXT, ts DATE, close NUMERIC, PRIMARY KEY (ticker, ts))
news      (id PK, ts TIMESTAMPTZ, ticker TEXT, headline TEXT, sentiment NUMERIC)
```

### Compliance (seeded, served by `compliance-mcp`)

```
rules     (id PK, code TEXT UNIQUE, description TEXT, severity ENUM,
           regex_pattern TEXT NULL, applies_to ENUM[report,recommendation])
```

### Application

```
runs (
  id UUID PK, customer_id FK, status ENUM[queued,running,succeeded,failed],
  prompt_version TEXT, model_config JSONB,
  started_at, finished_at, total_tokens INT, total_cost_usd NUMERIC(10,4),
  report_md TEXT, report_json JSONB, error TEXT NULL
)

eval_results (
  id UUID PK, run_id FK, kind ENUM[code,judge,human],
  score NUMERIC, passed BOOL, payload JSONB, created_at
)

human_reviews (
  id UUID PK, run_id FK, reviewer TEXT, score INT, approved BOOL,
  notes TEXT, created_at
)

drift_events (
  id UUID PK, metric TEXT, baseline_window TSTZRANGE,
  current_window TSTZRANGE, statistic NUMERIC, p_value NUMERIC,
  psi NUMERIC, severity ENUM[info,warn,alert], created_at
)
```

### Embeddings

Conversation memory and report excerpts embedded with `nomic-embed-text` via
Ollama, stored in `pgvector` (`memory_embeddings(run_id, role, text, embedding
VECTOR(768))`).

---

## 6. API Design

FastAPI service on `:8000`. JSON in/out. No auth in the demo (single-user); a
`X-User` header is read and propagated to OTel attributes for traceability.

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/runs` | Body: `{customer_id, prompt_version?, model_config_id?}` → 202 `{run_id}` |
| `GET`  | `/runs/{id}` | Run status + report (if finished) + eval summary |
| `GET`  | `/runs` | Paginated list with filters (status, customer_id, since) |
| `GET`  | `/runs/{id}/trace` | Returns the trace tree (assembled from OTel exporter for the UI) |
| `POST` | `/runs/{id}/reviews` | Body: `{reviewer, score, approved, notes}` |
| `GET`  | `/customers` | List of demo customers for the UI dropdown |
| `GET`  | `/configs/models` | Available model configurations |
| `POST` | `/bench` | Body: `{dataset_id, config_ids[]}` → MLflow experiment id |
| `GET`  | `/drift` | Recent drift events |
| `GET`  | `/healthz`, `/readyz` | Liveness/readiness for compose healthchecks |

Errors follow RFC 7807 (`application/problem+json`). All responses include
`traceparent` so users can jump from API response to Jaeger.

---

## 7. Testing Strategy

Coverage target: **≥85% line coverage** on `app/`, **100%** on prompt-rendering
and eval-scoring helpers (pure functions). Tests are split into three tiers
that map directly to GitHub Actions jobs.

### Unit tests (`tests/unit/`)

- Prompt rendering: snapshot tests on Jinja-rendered prompts for each agent
- Pydantic models: round-trip validation, schema regression
- Eval scoring functions: deterministic inputs → known outputs
- MCP client wrapper: error mapping, timeout behavior, retries
- Drift statistics: known distributions → expected KS / PSI values
- LangGraph node functions: each tested with mocked LLM client and stub MCP

### Integration tests (`tests/integration/`)

Run against real Postgres, real MCP servers, and a **fake LLM** (records and
replays completions from `tests/fixtures/llm_cassettes/`).

- Full graph run for a fixture customer produces a schema-valid report
- Compliance loop: when a recommendation contains forbidden text, the writer
  is re-invoked at most twice, then the run is marked `failed` cleanly
- Checkpoint resume: kill the worker mid-run, restart, verify completion
- MCP tool failure: `customer-mcp` returns 500 → run completes with explicit
  error in compliance section, not a stack trace
- Eval pipeline: a run produces ≥1 row in each of `eval_results.kind`

### End-to-end smoke (`tests/e2e/`)

One test that brings up the compose stack in CI (`docker compose -f
compose.ci.yaml up -d`), POSTs a run, polls until done, asserts the report
exists and traces are visible via Jaeger's API. Runs on PRs but skippable
locally with `pytest -m "not e2e"`.

### Eval-as-tests

`tests/evals/` holds a **golden dataset** of 20 customer scenarios with
expected report properties (e.g. "must mention currency conversion fees" for
customer X). The `eval-smoke` CI job runs the suite against the default model
config; failures block merge. Full benchmarking (all model variants) is a
manual workflow_dispatch job.

---

## 8. Infrastructure & Deployment

Everything runs locally with `docker compose up`. No cloud account required.

### Compose services

| Service | Image / build | Notes |
|---------|---------------|-------|
| `app` | local build | FastAPI + LangGraph workers + Streamlit (multi-process via `honcho`) |
| `customer-mcp` | local build | MCP server; HTTP transport on `:7001` |
| `market-mcp` | local build | MCP server on `:7002` |
| `compliance-mcp` | local build | MCP server on `:7003` |
| `postgres` | `pgvector/pgvector:pg16` | Volumes for data; healthcheck on `pg_isready` |
| `ollama` | `ollama/ollama:latest` | Pulls `llama3.1:8b` and `nomic-embed-text` on first start via init container |
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.16.0` | sqlite backend, file artifact store on a volume |
| `otel-collector` | `otel/opentelemetry-collector:0.108.0` | OTLP in, exports to Jaeger + Prometheus |
| `jaeger` | `jaegertracing/all-in-one:1.60` | UI on `:16686` |
| `prometheus` | `prom/prometheus:v2.54.0` | Scrapes app + collector |
| `grafana` | `grafana/grafana:11.1.0` | Pre-provisioned dashboard + Prometheus datasource |
| `drift-monitor` | local build (same image as app, different command) | Cron-style loop |

`docker compose up` brings the whole stack to ready in <3 minutes on a typical
laptop (dominated by the first-time Ollama model pull, which is cached after).

### Configuration

A single `.env.example` is committed. The default values run everything
locally with Ollama. Switching to a hosted LLM is one variable change:
`LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini OPENAI_API_KEY=…`. The job-listed
multi-cloud breadth is demonstrated through this provider abstraction rather
than by provisioning real cloud resources.

### CI (`.github/workflows/`)

| Workflow | Trigger | Jobs |
|----------|---------|------|
| `ci.yaml` | push, PR | lint (ruff), type (mypy), unit, integration, eval-smoke, build-image |
| `e2e.yaml` | PR with `e2e` label, nightly | brings up compose stack, runs `tests/e2e/` |
| `bench.yaml` | manual `workflow_dispatch` | full model benchmark, posts MLflow comparison summary as PR comment |

---

## 9. Project Structure

```
.
├── README.md                    # Quickstart + architecture diagram + screenshots
├── compose.yaml                 # Local dev stack
├── compose.ci.yaml              # Slimmed-down stack for CI e2e
├── Dockerfile                   # Single image for app + workers + drift-monitor
├── pyproject.toml               # Hatch build, ruff/mypy/pytest config
├── .env.example
├── docs/
│   ├── PRD.md                   # This document
│   ├── architecture.md          # Deeper component walkthrough
│   ├── evals.md                 # How to add eval cases
│   └── runbook.md               # Operating the stack, common failures
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app factory
│   ├── settings.py              # Pydantic BaseSettings, env loading
│   ├── api/
│   │   ├── runs.py
│   │   ├── reviews.py
│   │   ├── bench.py
│   │   └── health.py
│   ├── agents/
│   │   ├── graph.py             # LangGraph StateGraph definition
│   │   ├── state.py             # Shared TypedDict / Pydantic state
│   │   ├── coordinator.py
│   │   ├── transactions.py
│   │   ├── portfolio.py
│   │   ├── market.py
│   │   ├── compliance.py
│   │   └── writer.py
│   ├── mcp/
│   │   ├── client.py            # Shared MCP client wrapper with OTel + retries
│   │   └── servers/
│   │       ├── customer/        # FastMCP-based server, own entrypoint
│   │       ├── market/
│   │       └── compliance/
│   ├── llm/
│   │   ├── provider.py          # LiteLLM-backed router; supports Ollama/OpenAI/Anthropic/Bedrock
│   │   ├── prompts/             # Jinja2 templates, versioned subfolders (v1/, v2/)
│   │   └── cost.py              # Per-model cost table + token accounting
│   ├── memory/
│   │   ├── checkpointer.py      # Postgres LangGraph checkpointer
│   │   └── embeddings.py        # pgvector embed + retrieve
│   ├── eval/
│   │   ├── code.py              # Schema + rule asserts
│   │   ├── judge.py             # LLM-as-judge runner
│   │   ├── human.py             # Review queue API
│   │   ├── runner.py            # Orchestrates all three, logs to MLflow
│   │   └── datasets/
│   │       └── golden.jsonl
│   ├── drift/
│   │   ├── monitor.py           # Cron loop entrypoint
│   │   └── stats.py             # KS, PSI implementations
│   ├── observability/
│   │   ├── otel.py              # SDK setup, exporters, instrumentation hooks
│   │   ├── metrics.py           # Custom counters/histograms
│   │   └── logging.py           # Structured JSON logging with trace correlation
│   └── ui/
│       ├── streamlit_app.py     # Entry: pages = ["Run", "Reports", "Trace", "Reviews", "Bench", "Drift"]
│       └── components/
├── configs/
│   ├── models.yaml              # Model variants for benchmarking
│   ├── prompts.yaml             # Active prompt versions per agent
│   └── grafana/
│       ├── dashboards/finsight.json
│       └── datasources/prometheus.yaml
├── db/
│   ├── migrations/              # Alembic migrations
│   └── seeds/                   # Fixture customers, accounts, transactions, prices, rules
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── llm_cassettes/       # Recorded LLM responses for deterministic tests
│   │   └── customers.json
│   ├── unit/
│   ├── integration/
│   ├── evals/
│   └── e2e/
├── scripts/
│   ├── seed_db.py
│   ├── pull_models.sh           # Ollama model pre-pull
│   └── run_bench.py
└── .github/
    └── workflows/
        ├── ci.yaml
        ├── e2e.yaml
        └── bench.yaml
```

### Module organization principles

- **Agents are thin** — each agent module is prompt + tool list + output schema +
  one node function. Business logic lives in `mcp/servers/` (the data layer)
  and `eval/` (the quality layer)
- **No agent imports another agent** — they communicate only through the
  LangGraph state, which keeps the graph reorderable
- **Provider abstractions are honest** — `llm/provider.py` and `mcp/client.py`
  are the only files that talk to external services; everything else takes them
  as dependencies
- **Observability is not optional** — `observability/otel.py` is initialized in
  `main.py` before anything else; agent nodes get a span automatically via a
  decorator from `observability/metrics.py`

---

## Out of scope (explicit non-goals)

- Real customer authentication or per-tenant isolation (single-user demo)
- Production deployment manifests for AgentCore / Kubernetes / Terraform
- A bespoke frontend framework — Streamlit is sufficient and the role is not a
  frontend role
- Java or C# microservices — listed in the job as language alternatives, not
  requirements; demonstrating the same patterns twice would dilute the project
- Real market data feeds — fixtures are realistic but static; the architecture
  supports swapping the market MCP server for a live one
