"""Drift events endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/drift")
async def get_drift(metric: str | None = None) -> JSONResponse:
    from sqlalchemy import select

    from app.memory.database import get_db_session
    from app.memory.models import DriftEvent

    async with get_db_session() as session:
        q = select(DriftEvent).order_by(DriftEvent.created_at.desc()).limit(100)
        if metric:
            q = q.where(DriftEvent.metric == metric)
        events = (await session.execute(q)).scalars().all()

    return JSONResponse(
        {
            "items": [
                {
                    "id": str(e.id),
                    "metric": e.metric,
                    "severity": e.severity,
                    "statistic": float(e.statistic) if e.statistic is not None else None,
                    "p_value": float(e.p_value) if e.p_value is not None else None,
                    "psi": float(e.psi) if e.psi is not None else None,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in events
            ]
        }
    )
