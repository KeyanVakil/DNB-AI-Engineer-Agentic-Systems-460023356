"""MCP client wrapper with OTel instrumentation and error normalisation.

Agents call `call_tool(server_name, tool_name, arguments)` via this wrapper;
they never reach out to MCP servers directly.
"""

from __future__ import annotations

import time
from typing import Any

from app.mcp.errors import ToolError
from app.observability.otel import get_tracer

_registry: dict[str, Any] | None = None


def get_registry() -> dict[str, Any]:
    """FastAPI dependency — replaced in tests via dependency_overrides."""
    global _registry
    if _registry is None:
        from app.settings import get_settings

        settings = get_settings()
        _registry = _build_http_registry(settings)
    return _registry


def _build_http_registry(settings: Any) -> dict[str, Any]:
    """Build HTTP-backed MCP server wrappers for production."""
    from app.mcp.http_server import HttpMCPServer

    return {
        "customer-mcp": HttpMCPServer("customer-mcp", settings.customer_mcp_url),
        "market-mcp": HttpMCPServer("market-mcp", settings.market_mcp_url),
        "compliance-mcp": HttpMCPServer("compliance-mcp", settings.compliance_mcp_url),
    }


class MCPClient:
    """Thin orchestration layer over the registry with OTel spans and error mapping."""

    def __init__(self, registry: dict[str, Any]) -> None:
        self._registry = registry

    async def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        server = self._registry[server_name]
        result = server.list_tools()
        if hasattr(result, "__await__"):
            return await result
        return result  # type: ignore[return-value]

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        tracer = get_tracer()
        span_name = f"tool.mcp.{server_name}.{tool_name}"
        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("mcp.server", server_name)
            span.set_attribute("mcp.tool", tool_name)
            t0 = time.perf_counter()
            try:
                server = self._registry[server_name]
                result = server.call_tool(tool_name, arguments)
                if hasattr(result, "__await__"):
                    result = await result
                elapsed_ms = (time.perf_counter() - t0) * 1000
                span.set_attribute("mcp.duration_ms", elapsed_ms)
                _record_tool_duration(server_name, tool_name, elapsed_ms / 1000)
                return result
            except Exception as exc:
                span.record_exception(exc)
                raise ToolError(server=server_name, tool=tool_name, cause=exc) from exc


def _record_tool_duration(server: str, tool: str, seconds: float) -> None:
    try:
        from app.observability.metrics import tool_call_duration_seconds

        tool_call_duration_seconds.labels(server=server, tool=tool).observe(seconds)
    except Exception:
        pass
