"""Compliance revision loop (PRD Testing Strategy + F1).

From the PRD:
- Compliance can request revisions (loop back) or approve
- When a recommendation contains forbidden text, the writer is re-invoked
  at most twice, then the run is marked `failed` cleanly
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


async def test_compliance_approve_short_circuits_revision(
    db_engine, fake_mcp_registry, customer_id
):
    from app.agents.graph import build_graph
    from tests.integration.test_graph import _multi_cassette_llm

    llm = _multi_cassette_llm({
        "coordinator": "coordinator",
        "transactions": "transactions",
        "portfolio": "portfolio",
        "market": "market",
        "compliance": "compliance",  # approves
        "writer": "writer",
    })
    graph = build_graph(llm=llm, mcp=fake_mcp_registry)

    final = await graph.ainvoke({"customer_id": customer_id})

    assert final["revision_count"] == 0
    assert final["report_json"]["compliance_notes"]


async def test_compliance_revise_loops_back_to_writer_once_then_succeeds(
    db_engine, fake_mcp_registry, customer_id
):
    """First compliance call demands a revision; second approves."""
    from app.agents.graph import build_graph
    from tests.conftest import FakeLLM

    class _ComplianceFlip:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, **kwargs):
            self.calls += 1
            cassette = "compliance_revise" if self.calls == 1 else "compliance"
            inner = FakeLLM(cassette=cassette, strict=False)
            return await inner.complete(**kwargs)

    compliance_llm = _ComplianceFlip()

    class _Router:
        async def complete(self, *, agent=None, **kwargs):
            if agent == "compliance":
                return await compliance_llm.complete(**kwargs)
            fallback = FakeLLM(cassette=agent or "default", strict=False)
            return await fallback.complete(**kwargs)

    graph = build_graph(llm=_Router(), mcp=fake_mcp_registry)
    final = await graph.ainvoke({"customer_id": customer_id})

    assert final["revision_count"] == 1
    assert final["status"] == "succeeded"
    assert compliance_llm.calls == 2


async def test_compliance_fails_run_after_two_failed_revisions(
    db_engine, fake_mcp_registry, customer_id
):
    """If compliance refuses three times in a row, the graph must stop and
    mark the run `failed` — no infinite loop."""
    from app.agents.graph import build_graph
    from tests.conftest import FakeLLM

    class _AlwaysRevise:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, **kwargs):
            self.calls += 1
            return await FakeLLM(cassette="compliance_revise", strict=False).complete(**kwargs)

    compliance_llm = _AlwaysRevise()

    class _Router:
        async def complete(self, *, agent=None, **kwargs):
            if agent == "compliance":
                return await compliance_llm.complete(**kwargs)
            fallback = FakeLLM(cassette=agent or "default", strict=False)
            return await fallback.complete(**kwargs)

    graph = build_graph(llm=_Router(), mcp=fake_mcp_registry)
    final = await graph.ainvoke({"customer_id": customer_id})

    assert final["status"] == "failed"
    assert final["revision_count"] <= 2
    assert "compliance" in (final.get("error") or "").lower()


async def test_compliance_violation_recorded_in_eval_results(
    client, customer_id, poll_until, fake_compliance_mcp
):
    """An approved run with compliance notes still produces a code eval row."""
    response = await client.post("/runs", json={"customer_id": customer_id})
    run_id = response.json()["run_id"]

    final = await poll_until(
        client, run_id, lambda r: r["status"] in ("succeeded", "failed"), timeout=20.0
    )
    assert final["status"] == "succeeded"
    assert final["evals"]["code"]["passed"] in (True, False)
