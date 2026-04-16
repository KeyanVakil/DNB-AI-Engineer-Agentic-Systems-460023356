"""Prometheus metrics for FinSight Agents."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

agent_step_duration_seconds = Histogram(
    "agent_step_duration_seconds",
    "Duration of each LangGraph agent step",
    labelnames=["agent"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total LLM tokens consumed",
    labelnames=["kind", "model"],
)

llm_cost_usd_total = Counter(
    "llm_cost_usd_total",
    "Total estimated LLM cost in USD",
    labelnames=["model"],
)

tool_call_duration_seconds = Histogram(
    "tool_call_duration_seconds",
    "Duration of MCP tool calls",
    labelnames=["server", "tool"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0),
)

eval_judge_score = Gauge(
    "eval_judge_score",
    "Latest LLM-judge evaluation score",
    labelnames=["dimension"],
)

finsight_drift_psi = Gauge(
    "finsight_drift_psi",
    "Population Stability Index for each tracked metric",
    labelnames=["metric"],
)

finsight_drift_ks_p = Gauge(
    "finsight_drift_ks_p",
    "KS test p-value for each tracked metric",
    labelnames=["metric"],
)
