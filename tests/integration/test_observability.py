"""Observability tests (Feature F5).

Acceptance criteria:
- OTel spans cover HTTP request, each LangGraph node, each MCP tool call,
  each LLM completion (prompt hash + token counts as attributes)
- Spans exportable (tested against an in-memory exporter — Jaeger in prod)
- Prometheus metrics: `agent_step_duration_seconds`,
  `llm_tokens_total{kind=in|out,model=…}`, `llm_cost_usd_total`,
  `tool_call_duration_seconds`, `eval_judge_score`
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


async def test_http_request_emits_root_span(client, customer_id, otel_spans):
    await client.post("/runs", json={"customer_id": customer_id})

    request_spans = [s for s in otel_spans if s["kind"] == "server" and s["name"].startswith("POST")]
    assert request_spans, f"no root HTTP span found: {[s['name'] for s in otel_spans]}"
    assert "http.route" in request_spans[0]["attributes"]


async def test_each_langgraph_node_gets_a_span(completed_run, otel_spans):
    names = {s["name"] for s in otel_spans}

    for expected in (
        "agent.coordinator",
        "agent.transactions",
        "agent.portfolio",
        "agent.market",
        "agent.compliance",
        "agent.writer",
    ):
        assert expected in names, f"missing span {expected!r}"


async def test_llm_completion_span_has_prompt_hash_and_token_counts(
    completed_run, otel_spans
):
    llm_spans = [s for s in otel_spans if s["name"] == "llm.completion"]
    assert llm_spans

    sample = llm_spans[0]
    attrs = sample["attributes"]
    assert "llm.prompt_hash" in attrs
    assert "llm.tokens.in" in attrs
    assert "llm.tokens.out" in attrs
    assert "llm.model" in attrs


async def test_mcp_tool_call_emits_span_with_server_and_tool_attrs(
    completed_run, otel_spans
):
    tool_spans = [s for s in otel_spans if s["name"].startswith("tool.mcp.")]
    assert tool_spans
    for span in tool_spans:
        assert "mcp.server" in span["attributes"]
        assert "mcp.tool" in span["attributes"]
        assert "mcp.duration_ms" in span["attributes"]


async def test_prometheus_metrics_endpoint_exposes_expected_series(client, completed_run):
    response = await client.get("/metrics")
    assert response.status_code == 200

    body = response.text
    required_metrics = (
        "agent_step_duration_seconds",
        "llm_tokens_total",
        "llm_cost_usd_total",
        "tool_call_duration_seconds",
        "eval_judge_score",
    )
    for metric in required_metrics:
        assert metric in body, f"Prometheus series {metric!r} not exposed"


async def test_llm_tokens_metric_is_labelled_by_kind_and_model(client, completed_run):
    body = (await client.get("/metrics")).text
    assert 'llm_tokens_total{kind="in"' in body or 'kind="in"' in body
    assert 'llm_tokens_total{kind="out"' in body or 'kind="out"' in body
    assert 'model=' in body


async def test_logs_include_trace_correlation(completed_run, capsys):
    """JSON logs emitted during a run must include `trace_id` and `span_id`."""
    from app.observability.logging import get_logger

    log = get_logger("test")
    log.info("sample event", extra={"run_id": completed_run["run_id"]})

    captured = capsys.readouterr().err or capsys.readouterr().out
    assert "trace_id" in captured
    assert "span_id" in captured


async def test_span_tree_accessible_via_api(client, completed_run):
    response = await client.get(f"/runs/{completed_run['run_id']}/trace")

    assert response.status_code == 200
    tree = response.json()
    assert tree["trace_id"]
    assert tree["children"]
