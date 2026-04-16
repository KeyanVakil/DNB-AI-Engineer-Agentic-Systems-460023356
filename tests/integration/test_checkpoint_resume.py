"""Checkpoint-based resume test (Feature F1 acceptance criterion).

"Killing the worker mid-run and restarting resumes from the last completed node."
We simulate the kill by raising a `WorkerKilled` exception inside a chosen node,
then re-running the graph with the same thread id and asserting that earlier
nodes are not re-executed (their tool call counts stay flat).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class _Killed(RuntimeError):
    pass


async def test_resume_from_last_completed_node(
    db_engine, fake_mcp_registry, fake_customer_mcp, customer_id
):
    from app.agents.graph import build_graph
    from tests.integration.test_graph import _multi_cassette_llm

    thread_id = "t-resume-1"
    llm = _multi_cassette_llm({
        name: name
        for name in ("coordinator", "transactions", "portfolio", "market", "compliance", "writer")
    })

    # First run: die inside the writer node, after all specialists + compliance
    graph = build_graph(llm=llm, mcp=fake_mcp_registry, thread_id=thread_id)
    graph.nodes["writer"].inject_fault(_Killed("oom"))

    with pytest.raises(_Killed):
        await graph.ainvoke({"customer_id": customer_id})

    pre_resume_customer_calls = len(fake_customer_mcp.calls)
    assert pre_resume_customer_calls > 0

    # Second run with the same thread id: specialists should NOT re-execute
    graph2 = build_graph(llm=llm, mcp=fake_mcp_registry, thread_id=thread_id)
    final = await graph2.ainvoke({"customer_id": customer_id})

    post_resume_customer_calls = len(fake_customer_mcp.calls)
    assert post_resume_customer_calls == pre_resume_customer_calls, (
        "customer-mcp was called again on resume — specialists were not skipped"
    )
    assert final["report_json"]["customer_id"] == customer_id


async def test_resume_uses_same_run_id_via_api(client, customer_id, poll_until):
    """End-to-end resume via the HTTP layer: kill the worker, restart, same run_id."""
    response = await client.post("/runs", json={"customer_id": customer_id})
    run_id = response.json()["run_id"]

    # Tell the orchestrator to simulate a worker death inside the writer
    await client.post(
        f"/runs/{run_id}/_test/inject_fault",
        json={"node": "writer", "error": "worker killed"},
    )
    failed = await poll_until(
        client, run_id, lambda r: r["status"] == "failed", timeout=10.0
    )
    assert failed["error"]

    # Explicitly resume
    resume = await client.post(f"/runs/{run_id}/resume")
    assert resume.status_code == 202

    final = await poll_until(
        client, run_id, lambda r: r["status"] == "succeeded", timeout=15.0
    )
    assert final["run_id"] == run_id
    assert final["report"]["markdown"]
