"""Market MCP server — HTTP transport on :7002."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="market-mcp")

TOOLS = [
    {"name": "get_prices", "description": "Get historical close prices for a ticker"},
    {"name": "get_indices", "description": "Get current index values"},
    {"name": "get_news", "description": "Get recent news for a ticker"},
]


@app.get("/tools")
async def list_tools() -> JSONResponse:
    return JSONResponse({"tools": TOOLS})


@app.post("/tools/get_prices")
async def get_prices(body: dict) -> JSONResponse:
    from app.memory.database import get_db_session
    from app.memory.models import Price
    from sqlalchemy import select

    ticker = body["ticker"]
    async with get_db_session() as session:
        rows = (
            await session.execute(
                select(Price).where(Price.ticker == ticker).order_by(Price.ts.desc()).limit(body.get("days", 30))
            )
        ).scalars().all()
    return JSONResponse([{"ticker": p.ticker, "ts": str(p.ts), "close": str(p.close)} for p in rows])


@app.post("/tools/get_indices")
async def get_indices(body: dict) -> JSONResponse:
    return JSONResponse([{"code": "OSEBX", "value": "1420.10", "change_pct": "0.8"}])


@app.post("/tools/get_news")
async def get_news(body: dict) -> JSONResponse:
    from app.memory.database import get_db_session
    from app.memory.models import News
    from sqlalchemy import select

    ticker = body["ticker"]
    limit = body.get("limit", 5)
    async with get_db_session() as session:
        rows = (
            await session.execute(
                select(News).where(News.ticker == ticker).order_by(News.ts.desc()).limit(limit)
            )
        ).scalars().all()
    return JSONResponse([
        {"id": n.id, "ts": n.ts.isoformat(), "ticker": n.ticker,
         "headline": n.headline, "sentiment": float(n.sentiment)} for n in rows
    ])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7002)
