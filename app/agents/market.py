"""Market researcher agent."""

from __future__ import annotations

from typing import Any

from app.agents.base import _TimedSpan, list_mcp_tools, call_mcp, parse_json_response, run_llm
from app.mcp.errors import ToolError


async def run(
    state: dict[str, Any],
    llm: Any,
    mcp: dict[str, Any],
) -> dict[str, Any]:
    errors: list[ToolError] = []

    with _TimedSpan("market"):
        tools = await list_mcp_tools(mcp, "market-mcp")
        tool_names = {t["name"] for t in tools}

        indices: list[dict[str, Any]] = []
        news: list[dict[str, Any]] = []

        try:
            if "get_indices" in tool_names:
                indices = await call_mcp(mcp, "market-mcp", "get_indices", {})
        except ToolError as e:
            errors.append(e)

        holdings = state.get("portfolio_result", {}).get("holdings", [])
        tickers = [h.get("ticker") for h in holdings if h.get("ticker")]

        for ticker in tickers[:3]:
            try:
                if "get_news" in tool_names:
                    n = await call_mcp(mcp, "market-mcp", "get_news", {"ticker": ticker, "limit": 3})
                    news.extend(n if isinstance(n, list) else [])
            except ToolError:
                pass

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a market researcher. Summarise market conditions relevant to "
                    "the customer's portfolio. Return JSON: summary, signals (list), notes."
                ),
            },
            {
                "role": "user",
                "content": f"Indices: {indices}\nNews: {news}\nHoldings: {holdings}",
            },
        ]

        resp = await run_llm(llm, model=None, messages=messages, agent="market")
        result: dict[str, Any] = {}
        try:
            result = parse_json_response(resp["content"])
        except Exception:
            result = {"notes": resp.get("content", "")}

    return {"market_result": result, "errors": errors}
