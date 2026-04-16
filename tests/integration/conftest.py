"""Integration-suite-specific fixtures and helpers."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


@pytest.fixture()
async def started_run(client, customer_id) -> dict[str, Any]:
    """POST /runs and return the parsed response body."""
    response = await client.post("/runs", json={"customer_id": customer_id})
    assert response.status_code == 202, response.text
    return response.json()


async def _poll_until(
    client: Any,
    run_id: str,
    predicate,
    timeout: float = 30.0,
    interval: float = 0.1,
) -> dict[str, Any]:
    deadline = asyncio.get_event_loop().time() + timeout
    last: dict[str, Any] = {}
    while asyncio.get_event_loop().time() < deadline:
        response = await client.get(f"/runs/{run_id}")
        if response.status_code == 200:
            last = response.json()
            if predicate(last):
                return last
        await asyncio.sleep(interval)
    raise AssertionError(f"run {run_id} never satisfied predicate; last payload={last!r}")


@pytest.fixture()
def poll_until():
    """Fixture wrapping the polling helper so tests can await it directly."""
    return _poll_until


@pytest.fixture()
async def completed_run(client, started_run, poll_until) -> dict[str, Any]:
    """Poll a started run until it succeeds; fail the test if it errors."""
    run = await poll_until(
        client,
        started_run["run_id"],
        lambda r: r["status"] in ("succeeded", "failed"),
        timeout=20.0,
    )
    assert run["status"] == "succeeded", run
    return run


all = ("started_run", "completed_run", "poll_until")
