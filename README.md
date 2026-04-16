# FinSight Agents

> A production-grade multi-agent financial insight system demonstrating LangGraph orchestration, MCP tool integration, LLM evaluations, OpenTelemetry observability, and drift detection — runnable locally with `docker compose up`.

Built as a reference implementation for the [DNB AI Engineer — Agentic Systems](https://www.finn.no/job/ad/460023356) position.

---

## Skills Demonstrated

| Job Requirement | Implementation |
|---|---|
| Multi-agent application design | LangGraph StateGraph with coordinator + 4 specialists + writer |
| Enterprise system integration via MCP | Three MCP servers (customer-data, market-data, compliance-rules) |
| Evaluation frameworks | Code asserts + LLM-as-judge + human review queue, logged to MLflow |
| MLflow experiment tracking | Per-run eval metrics, prompt versioning, model benchmarking |
| OpenTelemetry observability | Spans on every agent step, tool call, LLM completion; metrics to Prometheus; Grafana dashboard |
| Drift detection | KS test + PSI on judge score distributions, surfaced as Prometheus gauge + DB events |
| Docker / CI | `docker compose up` full stack; GitHub Actions lint → type-check → unit → integration |
| Python 3.12 + FastAPI + Pydantic v2 | Async API, strict types, RFC 7807 error responses |
| Cost/latency/capability comparison | `/bench` endpoint + `finsight bench` CLI, all variants logged to MLflow |

---

## Architecture

```
Streamlit UI  ──HTTP──►  FastAPI :8000
                                │
                         LangGraph Graph
                         ├── coordinator
                         ├── transactions  ─┐
                         ├── portfolio      │  (parallel)
                         ├── market      ───┘
                         ├── compliance  (revision loop, max 2)
                         └── writer
                                │
              ┌─────────────────┼──────────────────┐
       customer-mcp:7001   market-mcp:7002   compliance-mcp:7003
              │
           Postgres (state + pgvector embeddings)
           Ollama  (LLM + embeddings, local, no API key)
           MLflow  (eval tracking)
           OTel Collector → Jaeger + Prometheus → Grafana
```

---

## Quick Start

```bash
cp .env.example .env
docker compose up
```

- **Streamlit UI**: http://localhost:8501  
- **API docs**: http://localhost:8000/docs  
- **Jaeger traces**: http://localhost:16686  
- **Grafana**: http://localhost:3000  
- **MLflow**: http://localhost:5000

On first start, Ollama pulls `llama3.1:8b` and `nomic-embed-text` (~5 GB). Subsequent starts are instant.

---

## Running Tests

```bash
# Install deps
pip install -e ".[dev]"

# Unit tests only (no Postgres needed)
pytest tests/unit/ -v

# Integration tests (requires Postgres)
export TEST_DATABASE_URL=postgresql+asyncpg://finsight:finsight@localhost:5432/finsight_test
pytest tests/integration/ -v -m "not e2e"

# Skip e2e (full compose stack)
pytest -m "not e2e"
```

---

## CLI

```bash
# Regression eval between two prompt versions
finsight eval --baseline v1 --candidate v2 \
  --dataset tests/fixtures/customers.json \
  --output report.json

# Benchmark model variants
finsight bench \
  --config configs/models.yaml \
  --dataset app/eval/datasets/golden.jsonl \
  --output bench.json
```

---

## Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.12 | Job requirement; ecosystem leader for LLM/agent work |
| Orchestration | LangGraph | Named explicitly in job description; stateful graph with checkpointing |
| Tool integration | MCP (Model Context Protocol) | Named explicitly; three real servers with documented schemas |
| LLM (default) | Ollama (`llama3.1:8b`) | Zero API keys; fully local; swap to any provider via one env var |
| API | FastAPI + Pydantic v2 | Async-first; type-safe; RFC 7807 error responses |
| Database | Postgres 16 + pgvector | LangGraph checkpointer + embeddings in one place |
| Eval tracking | MLflow | Named in job description |
| Observability | OpenTelemetry → Jaeger + Prometheus + Grafana | Named in job description |
| Drift detection | KS test + PSI (scipy) | Named in job description |
| UI | Streamlit | Fast to build; appropriate for a backend/AI role demo |
| CI | GitHub Actions | Named in job description |
