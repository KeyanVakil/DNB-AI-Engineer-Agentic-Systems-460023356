"""Integration tests for the `/runs` REST API (Feature F1).

Covers every acceptance criterion on the POST/GET run lifecycle:
- `POST /runs` returns 202 with a run id within 200ms (work happens async)
- `GET /runs/{id}` exposes the live status, final report, and eval summary
- `GET /runs` supports pagination + filtering
- `GET /runs/{id}/trace` returns a trace tree
- Errors follow RFC 7807 `application/problem+json`
"""

from __future__ import annotations

import time
import uuid

import pytest

pytestmark = pytest.mark.integration


async def test_post_runs_returns_202_with_run_id(client, customer_id) -> None:
    response = await client.post("/runs", json={"customer_id": customer_id})

    assert response.status_code == 202
    body = response.json()
    # run_id must be a parseable UUID
    uuid.UUID(body["run_id"])
    assert body["status"] == "queued"
    assert body["customer_id"] == customer_id


async def test_post_runs_is_async_and_responds_in_under_200ms(
    client, customer_id
) -> None:
    start = time.perf_counter()
    response = await client.post("/runs", json={"customer_id": customer_id})
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert response.status_code == 202
    # PRD F1 acceptance criterion: <200ms — work happens async
    assert elapsed_ms < 200, f"POST /runs took {elapsed_ms:.1f}ms"


async def test_post_runs_accepts_optional_prompt_and_model_config(
    client, customer_id
) -> None:
    response = await client.post(
        "/runs",
        json={
            "customer_id": customer_id,
            "prompt_version": "v2",
            "model_config_id": "llama3.1-8b",
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["prompt_version"] == "v2"
    assert body["model_config_id"] == "llama3.1-8b"


async def test_post_runs_unknown_customer_returns_problem_json(client) -> None:
    response = await client.post("/runs", json={"customer_id": "does-not-exist"})

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    # RFC 7807 required fields
    assert body["type"]
    assert body["title"]
    assert body["status"] == 404
    assert "customer" in body["detail"].lower()


async def test_post_runs_missing_customer_id_returns_422(client) -> None:
    response = await client.post("/runs", json={})

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")


async def test_get_run_returns_404_for_unknown_id(client) -> None:
    response = await client.get(f"/runs/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")


async def test_get_run_returns_full_payload_when_finished(completed_run) -> None:
    assert completed_run["status"] == "succeeded"
    assert completed_run["customer_id"]
    assert completed_run["report"]["markdown"].startswith("#")
    report_json = completed_run["report"]["json"]
    # PRD F1: report must contain each of these sections
    for key in (
        "spending_summary",
        "portfolio_summary",
        "market_context",
        "compliance_notes",
        "recommendations",
    ):
        assert key in report_json
    assert 2 <= len(report_json["recommendations"]) <= 5
    assert completed_run["evals"]  # summary per eval kind
    assert set(completed_run["evals"].keys()) >= {"code", "judge"}


async def test_get_run_trace_returns_tree(client, completed_run) -> None:
    response = await client.get(f"/runs/{completed_run['run_id']}/trace")

    assert response.status_code == 200
    tree = response.json()
    # Root span is the HTTP request; children are LangGraph nodes and tool calls
    assert tree["name"].startswith("POST")
    names = _flatten_span_names(tree)
    for expected in (
        "agent.coordinator",
        "agent.transactions",
        "agent.portfolio",
        "agent.market",
        "agent.compliance",
        "agent.writer",
    ):
        assert expected in names, f"missing span: {expected}"


async def test_list_runs_is_paginated(client, customer_id) -> None:
    for _ in range(3):
        await client.post("/runs", json={"customer_id": customer_id})

    response = await client.get("/runs", params={"limit": 2})
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 2
    assert body["total"] >= 3
    assert "next_cursor" in body


async def test_list_runs_filters_by_status(client, customer_id, completed_run) -> None:
    # After the completed_run fixture we have exactly one succeeded run in the DB
    response = await client.get("/runs", params={"status": "succeeded"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) >= 1
    assert all(item["status"] == "succeeded" for item in body["items"])


async def test_list_runs_filters_by_customer_id(client, seed_fixtures) -> None:
    first = seed_fixtures["customers"][0]["id"]
    second = seed_fixtures["customers"][1]["id"]
    for cid in (first, first, second):
        await client.post("/runs", json={"customer_id": cid})

    response = await client.get("/runs", params={"customer_id": first})

    assert response.status_code == 200
    body = response.json()
    assert all(item["customer_id"] == first for item in body["items"])
    assert len(body["items"]) >= 2


async def test_post_run_with_invalid_json_returns_problem(client) -> None:
    response = await client.post("/runs", content=b"{not json")

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")


async def test_response_includes_traceparent_for_jaeger_jump(
    client, customer_id
) -> None:
    response = await client.post("/runs", json={"customer_id": customer_id})

    assert response.status_code == 202
    assert "traceparent" in response.headers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_span_names(node: dict) -> list[str]:
    names = [node["name"]]
    for child in node.get("children", []):
        names.extend(_flatten_span_names(child))
    return names
