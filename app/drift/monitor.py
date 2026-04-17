"""Drift monitor — compares recent metric distributions to a rolling baseline."""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import uuid

from app.drift.stats import ks_test
from app.drift.stats import psi as psi_fn


async def run_once() -> None:
    """Run one drift detection cycle. Called every 5 minutes by the compose service."""
    now = dt.datetime.now(dt.UTC)
    baseline_start = now - dt.timedelta(hours=25)
    baseline_end = now - dt.timedelta(hours=1)
    current_start = now - dt.timedelta(hours=1)
    current_end = now

    from sqlalchemy import text

    from app.memory.database import get_db_session

    async with get_db_session() as session:
        # Fetch baseline judge scores (24h window)
        baseline_rows = await session.execute(
            text(
                "SELECT score FROM eval_results "
                "WHERE kind = 'judge' "
                "AND created_at >= :start AND created_at < :end"
            ),
            {"start": baseline_start, "end": baseline_end},
        )
        baseline = [float(r[0]) for r in baseline_rows]

        # Fetch current window (last hour)
        current_rows = await session.execute(
            text(
                "SELECT score FROM eval_results "
                "WHERE kind = 'judge' "
                "AND created_at >= :start AND created_at < :end"
            ),
            {"start": current_start, "end": current_end},
        )
        current = [float(r[0]) for r in current_rows]

        if not baseline or not current:
            return

        statistic, p_value = ks_test(baseline, current)
        psi_val = _compute_psi(baseline, current)

        severity = _severity(p_value, psi_val)

        event_id = uuid.uuid4()
        await session.execute(
            text(
                "INSERT INTO drift_events "
                "(id, metric, baseline_window, current_window, statistic, p_value, psi, "
                "severity, created_at) "
                "VALUES (:id, :metric, tstzrange(:bstart, :bend), tstzrange(:cstart, :cend), "
                ":stat, :pval, :psi, :sev, :now)"
            ),
            {
                "id": str(event_id),
                "metric": "judge_score",
                "bstart": baseline_start,
                "bend": baseline_end,
                "cstart": current_start,
                "cend": current_end,
                "stat": round(statistic, 6),
                "pval": round(p_value, 6),
                "psi": round(psi_val, 6),
                "sev": severity,
                "now": now,
            },
        )

        _update_prometheus(psi_val, p_value)


def _compute_psi(baseline: list[float], current: list[float]) -> float:
    n_bins = 10
    all_vals = baseline + current
    if not all_vals:
        return 0.0
    lo, hi = min(all_vals), max(all_vals)
    if lo == hi:
        return 0.0
    step = (hi - lo) / n_bins

    def to_bins(vals: list[float]) -> list[float]:
        counts = [0] * n_bins
        for v in vals:
            idx = min(int((v - lo) / step), n_bins - 1)
            counts[idx] += 1
        total = max(len(vals), 1)
        return [c / total for c in counts]

    return psi_fn(to_bins(baseline), to_bins(current))


def _severity(p_value: float, psi_val: float) -> str:
    if p_value < 0.01 or psi_val > 0.2:
        return "alert"
    if p_value < 0.05 or psi_val > 0.1:
        return "warn"
    return "info"


def _update_prometheus(psi_val: float, p_value: float) -> None:
    try:
        from app.observability.metrics import finsight_drift_ks_p, finsight_drift_psi

        finsight_drift_psi.labels(metric="judge_score").set(psi_val)
        finsight_drift_ks_p.labels(metric="judge_score").set(p_value)
    except Exception:
        pass


async def run_loop(interval_seconds: int = 300) -> None:
    while True:
        with contextlib.suppress(Exception):
            await run_once()
        await asyncio.sleep(interval_seconds)
