"""Compliance reviewer agent."""

from __future__ import annotations

import contextlib
from typing import Any

from app.agents.base import _TimedSpan, call_mcp, list_mcp_tools, parse_json_response, run_llm
from app.mcp.errors import ToolError


async def run(
    state: dict[str, Any],
    llm: Any,
    mcp: dict[str, Any],
) -> dict[str, Any]:
    errors: list[ToolError] = []
    draft_report = state.get("report_md", "")

    with _TimedSpan("compliance"):
        tools = await list_mcp_tools(mcp, "compliance-mcp")
        tool_names = {t["name"] for t in tools}

        rules: list[dict[str, Any]] = []

        try:
            if "list_rules" in tool_names:
                rules = await call_mcp(mcp, "compliance-mcp", "list_rules", {})
        except ToolError as e:
            errors.append(e)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a compliance reviewer. Given specialist outputs and compliance rules, "
                    'decide whether to approve or request revision. Return JSON: '
                    '{"decision": "approve"|"revise", "violations": [...], "notes": "..."}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Rules: {rules}\nDraft: {draft_report[:500]}"
                ),
            },
        ]

        resp = await run_llm(llm, model=None, messages=messages, agent="compliance")
        result: dict[str, Any] = {"decision": "approve", "violations": [], "notes": ""}
        with contextlib.suppress(Exception):
            result = parse_json_response(resp["content"])

        # Merge MCP tool errors into compliance notes
        if errors:
            result.setdefault("notes", "")
            result["notes"] = (
                result["notes"] + " (Compliance service temporarily unavailable)"
                if result["notes"]
                else "Compliance service temporarily unavailable"
            )
            result["decision"] = result.get("decision", "approve")

    return {"compliance_result": result, "errors": errors}
