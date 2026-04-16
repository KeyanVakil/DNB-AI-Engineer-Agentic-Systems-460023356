"""Report writer agent."""

from __future__ import annotations

from typing import Any

from app.agents.base import _TimedSpan, parse_json_response, run_llm


async def run(
    state: dict[str, Any],
    llm: Any,
    mcp: dict[str, Any],
) -> dict[str, Any]:
    customer_id: str = state["customer_id"]
    compliance_result = state.get("compliance_result", {})
    base_compliance_notes = compliance_result.get("notes", "No compliance issues detected.")

    # Detect compliance MCP errors from accumulated state errors
    mcp_errors = [
        e for e in state.get("errors", []) if "unavailable" in str(e).lower()
    ]

    with _TimedSpan("writer"):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a financial report writer. Synthesise the specialist analyses into "
                    "a complete report. Return JSON with keys: report_md (markdown string starting "
                    "with #), report_json (object with: spending_summary, portfolio_summary, "
                    "market_context, compliance_notes, recommendations (list of 2-5 strings))."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Transactions: {state.get('transaction_result', {})}\n"
                    f"Portfolio: {state.get('portfolio_result', {})}\n"
                    f"Market: {state.get('market_result', {})}\n"
                    f"Compliance notes: {base_compliance_notes}\n"
                    f"Customer ID: {customer_id}"
                ),
            },
        ]

        resp = await run_llm(llm, model=None, messages=messages, agent="writer")
        result: dict[str, Any] = {}
        try:
            result = parse_json_response(resp["content"])
        except Exception:
            result = {}

        txn = state.get("transaction_result", {})
        port = state.get("portfolio_result", {})
        mkt = state.get("market_result", {})

        default_report: dict[str, Any] = {
            "customer_id": customer_id,
            "spending_summary": (
                txn.get("notes") or f"Total spend NOK {txn.get('total_spend_nok', 'N/A')}."
            ),
            "portfolio_summary": port.get("notes") or "Portfolio data compiled.",
            "market_context": mkt.get("notes") or mkt.get("summary") or "Market data compiled.",
            "compliance_notes": base_compliance_notes,
            "recommendations": [
                "Review your spending patterns against your savings goals.",
                "Consider portfolio diversification to manage concentration risk.",
                "Consult a financial advisor for personalised investment advice.",
            ],
        }

        report_json: dict[str, Any] = result.get("report_json", {})
        # Fill any missing required fields from the fallback
        for key, val in default_report.items():
            if key not in report_json or not report_json[key]:
                report_json[key] = val

        # Always stamp the real customer_id
        report_json["customer_id"] = customer_id

        # Override compliance_notes when compliance MCP was unavailable
        if mcp_errors:
            report_json["compliance_notes"] = (
                "Compliance review was unable to complete. Manual review required."
            )

        report_md = result.get("report_md") or _build_default_md(report_json)

    return {
        "report_json": report_json,
        "report_md": report_md,
        "status": "succeeded",
    }


def _build_default_md(report_json: dict[str, Any]) -> str:
    recs = "\n".join(
        f"{i + 1}. {r}" for i, r in enumerate(report_json.get("recommendations", []))
    )
    return (
        f"# Financial Insight Report\n\n"
        f"## Spending Summary\n{report_json.get('spending_summary', '')}\n\n"
        f"## Portfolio Summary\n{report_json.get('portfolio_summary', '')}\n\n"
        f"## Market Context\n{report_json.get('market_context', '')}\n\n"
        f"## Compliance Notes\n{report_json.get('compliance_notes', '')}\n\n"
        f"## Recommendations\n{recs}\n"
    )
