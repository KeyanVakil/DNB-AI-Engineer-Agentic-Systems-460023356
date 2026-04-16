"""RFC 7807 error handling and edge cases across the API surface."""

from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.integration


async def test_all_error_responses_use_problem_json(client) -> None:
    response = await client.get(f"/runs/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    # RFC 7807 mandatory fields
    for field in ("type", "title", "status"):
        assert field in body
    assert isinstance(body["status"], int)


async def test_validation_error_lists_field_errors(client) -> None:
    response = await client.post(
        "/runs", json={"customer_id": "", "prompt_version": 123}
    )

    assert response.status_code == 422
    body = response.json()
    assert body["status"] == 422
    assert body["title"]
    assert "errors" in body or "invalid_params" in body


async def test_method_not_allowed_returns_problem(client) -> None:
    response = await client.put("/healthz")

    assert response.status_code == 405
    assert response.headers["content-type"].startswith("application/problem+json")


async def test_error_response_includes_traceparent(client) -> None:
    response = await client.get(f"/runs/{uuid.uuid4()}")

    assert response.status_code == 404
    assert "traceparent" in response.headers


async def test_internal_error_does_not_leak_stack_trace(
    client, customer_id, monkeypatch
) -> None:
    from app.api import runs as runs_api

    def boom(*_a, **_k):
        raise RuntimeError("DATABASE_URL=postgres://leaked:password@host/db")

    monkeypatch.setattr(runs_api, "create_run_record", boom, raising=True)

    response = await client.post("/runs", json={"customer_id": customer_id})

    assert response.status_code == 500
    body = response.json()
    assert "leaked:password" not in str(body)
    assert "Traceback" not in str(body)
    assert body["title"]


async def test_unauthorized_with_no_x_user_still_ok_for_demo(client, customer_id) -> None:
    """The PRD notes the demo is single-user with no auth; X-User is optional."""
    response = await client.post("/runs", json={"customer_id": customer_id})

    assert response.status_code == 202


async def test_x_user_header_is_attached_to_otel_spans(
    client, customer_id, otel_spans
) -> None:
    await client.post(
        "/runs", json={"customer_id": customer_id}, headers={"X-User": "ops@dnb.no"}
    )

    server_spans = [s for s in otel_spans if s["kind"] == "server"]
    assert server_spans
    assert any(s["attributes"].get("user.id") == "ops@dnb.no" for s in server_spans)
