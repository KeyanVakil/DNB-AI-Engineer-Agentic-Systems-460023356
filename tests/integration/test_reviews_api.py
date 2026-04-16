"""Human review endpoints (Feature F3 — review queue)."""

from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.integration


async def test_post_review_persists_record(client, completed_run, db_session):
    from sqlalchemy import text

    response = await client.post(
        f"/runs/{completed_run['run_id']}/reviews",
        json={
            "reviewer": "kv@dnb.no",
            "score": 4,
            "approved": True,
            "notes": "Report reads well, numbers check out.",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["run_id"] == completed_run["run_id"]
    assert body["score"] == 4
    assert body["approved"] is True

    rows = await db_session.execute(
        text("SELECT reviewer, score, approved FROM human_reviews WHERE run_id = :rid"),
        {"rid": completed_run["run_id"]},
    )
    persisted = rows.one()
    assert persisted.reviewer == "kv@dnb.no"
    assert persisted.score == 4
    assert persisted.approved is True


async def test_post_review_rejects_score_out_of_range(client, completed_run):
    response = await client.post(
        f"/runs/{completed_run['run_id']}/reviews",
        json={"reviewer": "x@y", "score": 99, "approved": True},
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")


async def test_post_review_on_unknown_run_returns_404(client):
    response = await client.post(
        f"/runs/{uuid.uuid4()}/reviews",
        json={"reviewer": "x", "score": 3, "approved": False, "notes": ""},
    )

    assert response.status_code == 404


async def test_review_creates_eval_results_row_with_kind_human(
    client, completed_run, db_session
):
    from sqlalchemy import text

    await client.post(
        f"/runs/{completed_run['run_id']}/reviews",
        json={"reviewer": "x", "score": 5, "approved": True, "notes": ""},
    )

    rows = await db_session.execute(
        text(
            "SELECT kind, score, passed FROM eval_results "
            "WHERE run_id = :rid AND kind = 'human'"
        ),
        {"rid": completed_run["run_id"]},
    )
    human_rows = rows.all()
    assert human_rows
    assert human_rows[0].kind == "human"
    assert human_rows[0].score == 5.0
    assert human_rows[0].passed is True


async def test_review_queue_excludes_already_reviewed_runs(client, completed_run):
    await client.post(
        f"/runs/{completed_run['run_id']}/reviews",
        json={"reviewer": "x", "score": 4, "approved": True, "notes": ""},
    )

    queue = (await client.get("/reviews/queue")).json()
    run_ids = [item["run_id"] for item in queue["items"]]

    assert completed_run["run_id"] not in run_ids
