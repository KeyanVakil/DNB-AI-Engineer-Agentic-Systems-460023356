"""Customer listing endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/customers")
async def get_customers() -> JSONResponse:
    from sqlalchemy import select

    from app.memory.database import get_db_session
    from app.memory.models import Customer

    async with get_db_session() as session:
        customers = (await session.execute(select(Customer))).scalars().all()

    return JSONResponse(
        {
            "items": [
                {"id": c.id, "name": c.name, "segment": c.segment}
                for c in customers
            ]
        }
    )
