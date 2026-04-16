"""Human review endpoints."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, field_validator

from app.api.errors import not_found, unprocessable

router = APIRouter()


class _ReviewBody(BaseModel):
    reviewer: str
    score: int
    approved: bool
    notes: str | None = None

    @field_validator("score")
    @classmethod
    def _valid_score(cls, v: int) -> int:
        if not (1 <= v <= 5):
            raise ValueError("score must be between 1 and 5")
        return v


@router.post("/runs/{run_id}/reviews", status_code=201)
async def post_review(run_id: str, request: Request) -> JSONResponse:
    try:
        uid = uuid.UUID(run_id)
    except ValueError:
        return not_found(f"Run not found: {run_id}")

    raw = await request.body()

    import json as _json

    try:
        data = _json.loads(raw)
    except Exception:
        from app.api.errors import bad_request
        return bad_request("Invalid JSON")

    try:
        body = _ReviewBody.model_validate(data)
    except ValidationError as exc:
        errors = [{"field": str(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return unprocessable("Validation error", errors)

    from app.memory.database import get_db_session
    from app.memory.models import EvalResult, HumanReview, Run
    from sqlalchemy import select

    async with get_db_session() as session:
        run = (await session.execute(select(Run).where(Run.id == uid))).scalar_one_or_none()
        if run is None:
            return not_found(f"Run {run_id} not found")

        review_id = uuid.uuid4()
        review = HumanReview(
            id=review_id,
            run_id=uid,
            reviewer=body.reviewer,
            score=body.score,
            approved=body.approved,
            notes=body.notes,
        )
        session.add(review)

        session.add(
            EvalResult(
                id=uuid.uuid4(),
                run_id=uid,
                kind="human",
                score=Decimal(str(body.score)),
                passed=body.approved,
                payload={"reviewer": body.reviewer, "notes": body.notes or ""},
            )
        )

    return JSONResponse(
        {
            "review_id": str(review_id),
            "run_id": run_id,
            "reviewer": body.reviewer,
            "score": body.score,
            "approved": body.approved,
        },
        status_code=201,
    )


@router.get("/reviews/queue")
async def get_review_queue() -> JSONResponse:
    from app.memory.database import get_db_session
    from app.memory.models import HumanReview, Run
    from sqlalchemy import select

    async with get_db_session() as session:
        reviewed_ids = {
            str(r.run_id)
            for r in (await session.execute(select(HumanReview))).scalars().all()
        }
        succeeded_runs = (
            await session.execute(select(Run).where(Run.status == "succeeded"))
        ).scalars().all()

        queue = [
            {"run_id": str(r.id), "customer_id": r.customer_id, "prompt_version": r.prompt_version}
            for r in succeeded_runs
            if str(r.id) not in reviewed_ids
        ]

    return JSONResponse({"items": queue})
