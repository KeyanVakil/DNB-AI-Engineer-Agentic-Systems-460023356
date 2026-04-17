"""Base helpers shared by all agent node functions."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any


async def call_mcp(
    registry: dict[str, Any],
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    """Call an MCP tool via the registry, wrapping in OTel span."""
    from app.mcp.client import MCPClient

    client = MCPClient(registry)
    return await client.call_tool(server_name, tool_name, arguments)


async def list_mcp_tools(registry: dict[str, Any], server_name: str) -> list[dict[str, Any]]:
    from app.mcp.client import MCPClient

    client = MCPClient(registry)
    return await client.list_tools(server_name)


def parse_json_response(content: str) -> dict[str, Any]:
    """Parse an LLM response as JSON, tolerating markdown fences."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


async def run_llm(
    llm: Any,
    *,
    model: Any,
    messages: list[dict[str, Any]],
    agent: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Call llm.complete() wrapped in an llm.completion OTel span.

    Works for any LLM provider — including FakeLLM in tests.
    """
    from app.observability.otel import get_tracer

    prompt_hash = hashlib.sha256(
        json.dumps(messages, sort_keys=True).encode()
    ).hexdigest()[:16]

    effective_model = model or "unknown"
    with get_tracer().start_as_current_span("llm.completion") as span:
        span.set_attribute("llm.model", effective_model)
        span.set_attribute("llm.prompt_hash", prompt_hash)
        resp = await llm.complete(model=model, messages=messages, agent=agent, **kwargs)
        usage = resp.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        span.set_attribute("llm.tokens.in", tokens_in)
        span.set_attribute("llm.tokens.out", tokens_out)
        try:
            from app.observability.metrics import llm_tokens_total
            llm_tokens_total.labels(kind="in", model=effective_model).inc(tokens_in)
            llm_tokens_total.labels(kind="out", model=effective_model).inc(tokens_out)
        except Exception:
            pass
    return resp


def agent_span(agent_name: str) -> Any:
    """Context manager that creates an agent.{name} OTel span."""
    from app.observability.otel import get_tracer

    return get_tracer().start_as_current_span(f"agent.{agent_name}")


class _TimedSpan:
    def __init__(self, agent_name: str) -> None:
        self._name = agent_name
        self._span_ctx: Any = None
        self._t0: float = 0.0

    def __enter__(self) -> _TimedSpan:
        from app.observability.otel import get_tracer

        self._span_ctx = get_tracer().start_as_current_span(f"agent.{self._name}")
        self._span_ctx.__enter__()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = time.perf_counter() - self._t0
        try:
            from app.observability.metrics import agent_step_duration_seconds

            agent_step_duration_seconds.labels(agent=self._name).observe(elapsed)
        except Exception:
            pass
        if self._span_ctx:
            self._span_ctx.__exit__(*args)
