"""Drift detection tests (Feature F6).

Acceptance criteria:
- Drift job runs every 5 minutes (tested by invoking the job entrypoint)
- Compares last-hour judge scores + tool-call distributions to 24h baseline
  via KS test + PSI
- Drift events written to `drift_events` and exposed as Prometheus gauge
- UI exposes drift via `/drift` endpoint
- Thresholds: `psi > 0.2` or `ks_p < 0.01` -> alert severity
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import text

pytestmark = [pytest.mark.integration, pytest.mark.drift]


@pytest.fixture()
async def baseline_eval_scores(db_session):
    """Seed 24h of stable judge scores for a baseline distribution."""
    from app.memory import seeds

    now = dt.datetime.now(dt.UTC)
    records = [
        {
            "run_id": None,
            "kind": "judge",
            "score": 4.0 + (i % 3) * 0.1,
            "passed": True,
            "payload": {},
            "created_at": now - dt.timedelta(hours=1 + i / 24),
        }
        for i in range(100)
    ]
    await seeds.load_eval_results(db_session, records)
    await db_session.commit()
    return records


@pytest.fixture()
async def shifted_eval_scores(db_session):
    """Seed last-hour scores with a clear downward shift (should trigger alert)."""
    from app.memory import seeds

    now = dt.datetime.now(dt.UTC)
    records = [
        {
            "run_id": None,
            "kind": "judge",
            "score": 2.0 + (i % 3) * 0.1,
            "passed": False,
            "payload": {},
            "created_at": now - dt.timedelta(minutes=i),
        }
        for i in range(30)
    ]
    await seeds.load_eval_results(db_session, records)
    await db_session.commit()
    return records


async def test_drift_monitor_detects_shift_and_writes_event(
    baseline_eval_scores, shifted_eval_scores, db_session
):
    from app.drift.monitor import run_once

    await run_once()

    rows = await db_session.execute(
        text("SELECT metric, statistic, p_value, psi, severity FROM drift_events")
    )
    events = rows.all()
    assert events, "drift monitor produced no events"

    judge_events = [e for e in events if e.metric == "judge_score"]
    assert judge_events
    alert = judge_events[0]
    assert alert.severity in ("warn", "alert")
    assert (alert.psi or 0) > 0.2 or (alert.p_value or 1) < 0.01


async def test_drift_monitor_no_event_when_distributions_match(
    baseline_eval_scores, db_session
):
    """When the current window looks like the baseline, no alert fires."""
    from app.drift.monitor import run_once

    await run_once()

    rows = await db_session.execute(
        text(
            "SELECT severity FROM drift_events "
            "WHERE metric = 'judge_score' AND severity = 'alert'"
        )
    )
    assert rows.all() == []


async def test_drift_endpoint_returns_recent_events(
    client, baseline_eval_scores, shifted_eval_scores
):
    from app.drift.monitor import run_once

    await run_once()

    response = await client.get("/drift")
    assert response.status_code == 200

    body = response.json()
    assert body["items"]
    for event in body["items"]:
        assert event["metric"]
        assert event["severity"] in ("info", "warn", "alert")
        assert "statistic" in event
        assert "created_at" in event


async def test_drift_endpoint_filters_by_metric(
    client, baseline_eval_scores, shifted_eval_scores
):
    from app.drift.monitor import run_once

    await run_once()

    response = await client.get("/drift", params={"metric": "judge_score"})
    assert response.status_code == 200
    body = response.json()
    assert all(item["metric"] == "judge_score" for item in body["items"])


async def test_drift_metric_exposed_to_prometheus(
    client, baseline_eval_scores, shifted_eval_scores
):
    from app.drift.monitor import run_once

    await run_once()
    body = (await client.get("/metrics")).text

    assert "finsight_drift_psi" in body or "drift_psi" in body


async def test_ks_statistic_function_against_known_distribution():
    from app.drift.stats import ks_test

    baseline = [1.0] * 100
    current = [5.0] * 100
    statistic, p_value = ks_test(baseline, current)

    assert statistic == pytest.approx(1.0, abs=0.01)
    assert p_value < 0.01


async def test_psi_function_against_identical_distribution():
    from app.drift.stats import psi

    a = [0.2, 0.3, 0.5]
    assert psi(a, a) == pytest.approx(0.0, abs=1e-6)


async def test_psi_function_flags_shift():
    from app.drift.stats import psi

    baseline_bins = [0.25, 0.25, 0.25, 0.25]
    shifted_bins = [0.7, 0.1, 0.1, 0.1]

    assert psi(baseline_bins, shifted_bins) > 0.2
