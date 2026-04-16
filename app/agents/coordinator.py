"""Coordinator agent — plans the analysis run."""

from __future__ import annotations

from typing import Any

from app.agents.base import _TimedSpan, run_llm


async def run(
    state: dict[str, Any],
    llm: Any,
    mcp: dict[str, Any],
) -> dict[str, Any]:
    customer_id: str = state["customer_id"]

    with _TimedSpan("coordinator"):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a financial insight coordinator. Plan the analysis for the "
                    "given customer. Always respond with JSON: "
                    '{\"plan\": [...], \"rationale\": \"...\"}'
                ),
            },
            {
                "role": "user",
                "content": f"Customer ID: {customer_id}. Plan the full analysis.",
            },
        ]

        await run_llm(llm, model=None, messages=messages, agent="coordinator")

    return {
        "customer_profile": {},
        "errors": [],
        "revision_count": state.get("revision_count", 0),
        "status": "running",
    }
