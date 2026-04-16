"""Liveness and readiness endpoints used by compose healthchecks."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


async def test_healthz_returns_200(client) -> None:
    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readyz_returns_200_when_dependencies_are_up(client, seed_fixtures) -> None:
    response = await client.get("/readyz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert set(body["checks"]) >= {"postgres", "customer-mcp", "market-mcp", "compliance-mcp"}
    for check in body["checks"].values():
        assert check["ok"] is True


async def test_readyz_returns_503_when_an_mcp_server_is_down(
    client, fake_customer_mcp
) -> None:
    fake_customer_mcp.fail_next = RuntimeError("connection refused")

    response = await client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["customer-mcp"]["ok"] is False


async def test_health_responses_include_traceparent(client) -> None:
    response = await client.get("/healthz")

    assert "traceparent" in response.headers
    # Matches the W3C traceparent header format: `00-<trace-id>-<span-id>-<flags>`
    parts = response.headers["traceparent"].split("-")
    assert len(parts) == 4
    assert parts[0] == "00"
