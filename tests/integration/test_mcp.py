"""MCP tool integration tests (Feature F2).

Acceptance criteria:
- Three MCP servers are running as separate processes in compose
- Each exposes ≥3 tools with documented schemas
- Agents discover tools via `list_tools`; names are not hardcoded
- Stubbing an MCP server at test time requires zero changes to agent code
- Failing tool calls surface as typed errors in graph state, not crashes
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


async def test_each_mcp_server_exposes_at_least_three_tools(
    fake_customer_mcp, fake_market_mcp, fake_compliance_mcp
):
    for server in (fake_customer_mcp, fake_market_mcp, fake_compliance_mcp):
        tools = await server.list_tools()
        assert len(tools) >= 3, f"{server.name} exposes {len(tools)} tools, expected >=3"
        for tool in tools:
            assert tool["name"]
            assert tool["description"]


async def test_agents_discover_tools_via_list_tools(
    fake_mcp_registry, fake_llm, customer_id
):
    """The transactions agent must not hardcode tool names — it should call
    `list_tools` at least once before invoking any tool."""
    from app.agents.transactions import run as run_transactions

    server = fake_mcp_registry["customer-mcp"]
    list_tools_calls = 0
    original_list_tools = server.list_tools

    async def counting_list_tools():
        nonlocal list_tools_calls
        list_tools_calls += 1
        return await original_list_tools()

    server.list_tools = counting_list_tools  # type: ignore[assignment]

    await run_transactions(
        state={"customer_id": customer_id}, llm=fake_llm, mcp=fake_mcp_registry
    )

    assert list_tools_calls >= 1


async def test_mcp_stubbing_requires_no_agent_code_changes(
    fake_mcp_registry, fake_llm, customer_id
):
    """Replace customer-mcp with a completely different stub and verify the
    transactions agent still runs — i.e. it reaches tools only via the registry."""
    from app.agents.transactions import run as run_transactions
    from tests.conftest import FakeMCPServer

    replacement = FakeMCPServer("customer-mcp")
    replacement.register("get_profile", lambda customer_id: {"id": customer_id, "name": "Stub"})
    replacement.register("get_accounts", lambda customer_id: [])
    replacement.register("get_transactions", lambda customer_id, days=30: [])
    replacement.register("get_holdings", lambda customer_id: [])
    fake_mcp_registry["customer-mcp"] = replacement

    result = await run_transactions(
        state={"customer_id": customer_id}, llm=fake_llm, mcp=fake_mcp_registry
    )

    assert result is not None
    assert replacement.calls, "agent never called the replacement server"


async def test_failing_tool_surfaces_as_typed_error_in_state(
    fake_mcp_registry, fake_llm, customer_id
):
    """A failing MCP tool must appear as a `ToolError` in the graph state,
    not an exception propagated out of the run."""
    from app.agents.transactions import run as run_transactions
    from app.mcp.errors import ToolError

    fake_mcp_registry["customer-mcp"].fail_next = ConnectionError("boom")

    result = await run_transactions(
        state={"customer_id": customer_id}, llm=fake_llm, mcp=fake_mcp_registry
    )

    assert "errors" in result
    errors = result["errors"]
    assert any(isinstance(err, ToolError) for err in errors)
    assert any(err.server == "customer-mcp" for err in errors)


async def test_full_run_continues_when_one_mcp_tool_fails(
    client, customer_id, fake_compliance_mcp, poll_until
):
    """When compliance-mcp returns 500, the run should STILL finish — the
    error is surfaced as an explicit compliance note, not a stack trace."""
    fake_compliance_mcp.fail_next = RuntimeError("503 Service Unavailable")

    response = await client.post("/runs", json={"customer_id": customer_id})
    run_id = response.json()["run_id"]
    final = await poll_until(
        client, run_id, lambda r: r["status"] in ("succeeded", "failed"), timeout=20.0
    )

    # Per the PRD: "run completes with explicit error in compliance section,
    # not a stack trace"
    assert final["status"] == "succeeded"
    compliance_notes = final["report"]["json"]["compliance_notes"].lower()
    assert "unavailable" in compliance_notes or "unable" in compliance_notes
    assert "traceback" not in compliance_notes


async def test_mcp_tool_schema_is_discoverable(fake_customer_mcp):
    tools = await fake_customer_mcp.list_tools()

    expected_tool_names = {"get_profile", "get_accounts", "get_transactions", "get_holdings"}
    actual = {t["name"] for t in tools}
    assert expected_tool_names.issubset(actual)


async def test_mcp_client_wraps_calls_in_otel_span(
    fake_mcp_registry, fake_llm, customer_id, otel_spans
):
    from app.agents.transactions import run as run_transactions

    await run_transactions(
        state={"customer_id": customer_id}, llm=fake_llm, mcp=fake_mcp_registry
    )

    tool_spans = [s for s in otel_spans if s["name"].startswith("tool.mcp.")]
    assert tool_spans, "no tool.mcp.* spans emitted"
    sample = tool_spans[0]
    assert "mcp.server" in sample["attributes"]
    assert "mcp.tool" in sample["attributes"]
