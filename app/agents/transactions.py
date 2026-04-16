"""Transaction analyst agent."""

from __future__ import annotations

from typing import Any

from app.agents.base import _TimedSpan, list_mcp_tools, call_mcp, parse_json_response, run_llm
from app.mcp.errors import ToolError


async def run(
    state: dict[str, Any],
    llm: Any,
    mcp: dict[str, Any],
) -> dict[str, Any]:
    customer_id: str = state["customer_id"]
    errors: list[ToolError] = []

    with _TimedSpan("transactions"):
        tools = await list_mcp_tools(mcp, "customer-mcp")
        tool_names = {t["name"] for t in tools}

        transactions: list[dict[str, Any]] = []

        try:
            if "get_transactions" in tool_names:
                transactions = await call_mcp(
                    mcp, "customer-mcp", "get_transactions", {"customer_id": customer_id, "days": 90}
                )
        except ToolError as e:
            errors.append(e)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a transaction analyst. Analyse the provided transactions and "
                    "return a JSON summary with keys: total_income_nok, total_spend_nok, "
                    "top_categories, recurring_items, notes."
                ),
            },
            {
                "role": "user",
                "content": f"Transactions: {transactions}",
            },
        ]

        resp = await run_llm(llm, model=None, messages=messages, agent="transactions")
        result: dict[str, Any] = {}
        try:
            result = parse_json_response(resp["content"])
        except Exception:
            result = {"notes": resp.get("content", "")}

    return {"transaction_result": result, "errors": errors}
