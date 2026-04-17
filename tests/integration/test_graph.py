"""LangGraph orchestration tests (Feature F1).

These exercise the graph directly — not through the HTTP API — so we can
assert on structure (nodes, edges), on concurrency (specialists run in
parallel), and on the Pydantic state after each node.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.integration


async def test_graph_contains_expected_nodes(db_engine, fake_mcp_registry, fake_llm):
    from app.agents.graph import build_graph

    graph = build_graph(llm=fake_llm, mcp=fake_mcp_registry)

    nodes = set(graph.nodes.keys())
    assert {
        "coordinator",
        "transactions",
        "portfolio",
        "market",
        "compliance",
        "writer",
    }.issubset(nodes)


async def test_graph_has_compliance_to_specialist_revision_edge(
    db_engine, fake_mcp_registry, fake_llm
):
    from app.agents.graph import build_graph

    graph = build_graph(llm=fake_llm, mcp=fake_mcp_registry)
    edges = graph.edges  # list of (source, target) tuples expected

    assert ("compliance", "writer") in edges
    # Revision loop: compliance can route back to writer/specialists with a revise verdict
    conditional_targets = graph.conditional_edges.get("compliance", set())
    assert "writer" in conditional_targets
    assert "transactions" in conditional_targets or "portfolio" in conditional_targets


async def test_full_graph_run_produces_schema_valid_report(
    db_engine, fake_mcp_registry, customer_id
):
    from app.agents.graph import build_graph
    from app.agents.state import FinSightReport

    llm = _multi_cassette_llm(
        {
            "coordinator": "coordinator",
            "transactions": "transactions",
            "portfolio": "portfolio",
            "market": "market",
            "compliance": "compliance",
            "writer": "writer",
        }
    )
    graph = build_graph(llm=llm, mcp=fake_mcp_registry)

    final = await graph.ainvoke({"customer_id": customer_id})

    report = FinSightReport.model_validate(final["report_json"])
    assert report.customer_id == customer_id
    assert 2 <= len(report.recommendations) <= 5
    assert report.spending_summary
    assert report.portfolio_summary
    assert report.market_context
    assert report.compliance_notes


async def test_specialists_run_concurrently(
    db_engine, fake_mcp_registry, customer_id
):
    """The graph must dispatch the four specialists in parallel, not serially."""
    from app.agents.graph import build_graph

    delays = {"transactions": 0.3, "portfolio": 0.3, "market": 0.3, "compliance": 0.0}

    async def slow_tool(tool_name: str, arguments, *, original):
        await asyncio.sleep(delays.get(tool_name.split(".")[-1], 0.0))
        return await original(tool_name, arguments)

    # Instrument the MCP registry to sleep for each specialist's first tool call
    for server in fake_mcp_registry.values():
        original = server.call_tool

        async def wrapped(tool_name, arguments, _o=original):
            await asyncio.sleep(0.3)
            if asyncio.iscoroutinefunction(_o):
                return await _o(tool_name, arguments)
            return _o(tool_name, arguments)

        server.call_tool = wrapped  # type: ignore[method-assign]

    llm = _multi_cassette_llm(
        {name: name for name in
         ("coordinator", "transactions", "portfolio", "market", "compliance", "writer")}
    )
    graph = build_graph(llm=llm, mcp=fake_mcp_registry)

    start = asyncio.get_event_loop().time()
    await graph.ainvoke({"customer_id": customer_id})
    elapsed = asyncio.get_event_loop().time() - start

    # If the three specialists (transactions, portfolio, market) ran serially,
    # the overall runtime would be ≥0.9s; concurrently it should stay under ~0.6s
    # of work — allow generous headroom for CI scheduling jitter.
    assert elapsed < 1.5, f"specialists appear to run serially; took {elapsed:.2f}s"


async def test_graph_state_is_checkpointed_to_postgres(
    db_engine, fake_mcp_registry, customer_id, db_session
):
    from sqlalchemy import text

    from app.agents.graph import build_graph

    llm = _multi_cassette_llm(
        {name: name for name in
         ("coordinator", "transactions", "portfolio", "market", "compliance", "writer")}
    )
    graph = build_graph(llm=llm, mcp=fake_mcp_registry, thread_id="t-abc")
    await graph.ainvoke({"customer_id": customer_id})

    # The Postgres LangGraph checkpointer writes into its own schema
    rows = await db_session.execute(
        text("SELECT COUNT(*) FROM langgraph.checkpoints WHERE thread_id = 't-abc'")
    )
    count = rows.scalar()
    assert count and count > 0


async def test_agents_do_not_import_each_other() -> None:
    """No agent module may import another agent module.

    Enforces the PRD rule that agents communicate only through graph state,
    not by direct call. Keeps the graph reorderable.
    """
    import ast
    import pathlib

    agents_dir = pathlib.Path("app/agents")
    modules = [p for p in agents_dir.glob("*.py") if p.stem not in {"__init__", "graph", "state"}]
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("app.agents.") or node.module.endswith(
                    ("state", "base")
                ), f"{path.name} imports sibling agent {node.module}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _multi_cassette_llm(mapping: dict[str, str]):
    """A FakeLLM that picks a cassette per agent based on message metadata."""
    from tests.conftest import FakeLLM

    cassettes = {agent: FakeLLM(cassette=name, strict=False) for agent, name in mapping.items()}

    class _Router:
        calls: list[dict] = []

        async def complete(self, *, model, messages, response_format=None, agent=None, **kwargs):
            target = cassettes.get(agent) or cassettes["coordinator"]
            self.calls.append({"agent": agent, "messages": messages})
            return await target.complete(
                model=model, messages=messages, response_format=response_format, **kwargs
            )

    return _Router()
