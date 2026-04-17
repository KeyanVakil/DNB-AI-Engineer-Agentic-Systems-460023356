"""Portfolio analyst agent."""

from __future__ import annotations

from typing import Any

from app.agents.base import _TimedSpan, call_mcp, list_mcp_tools, parse_json_response, run_llm
from app.mcp.errors import ToolError


async def run(
    state: dict[str, Any],
    llm: Any,
    mcp: dict[str, Any],
) -> dict[str, Any]:
    customer_id: str = state["customer_id"]
    errors: list[ToolError] = []

    with _TimedSpan("portfolio"):
        tools = await list_mcp_tools(mcp, "customer-mcp")
        tool_names = {t["name"] for t in tools}

        holdings: list[dict[str, Any]] = []

        try:
            if "get_holdings" in tool_names:
                holdings = await call_mcp(
                    mcp, "customer-mcp", "get_holdings", {"customer_id": customer_id}
                )
        except ToolError as e:
            errors.append(e)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a portfolio analyst. Analyse the holdings and return JSON with: "
                    "holdings (list), concentration_warning (bool), notes."
                ),
            },
            {
                "role": "user",
                "content": f"Holdings: {holdings}",
            },
        ]

        resp = await run_llm(llm, model=None, messages=messages, agent="portfolio")
        result: dict[str, Any] = {}
        try:
            result = parse_json_response(resp["content"])
        except Exception:
            result = {"notes": resp.get("content", ""), "holdings": holdings}

    return {"portfolio_result": result, "errors": errors}
