"""Customer MCP server — HTTP transport on :7001."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.memory.database import get_db_session
from app.memory.models import Account, Customer, Holding, Transaction
from sqlalchemy import select

app = FastAPI(title="customer-mcp")

TOOLS = [
    {"name": "get_profile", "description": "Get customer profile by ID"},
    {"name": "get_accounts", "description": "Get accounts for a customer"},
    {"name": "get_transactions", "description": "Get recent transactions for a customer"},
    {"name": "get_holdings", "description": "Get holdings for a customer"},
]


@app.get("/tools")
async def list_tools() -> JSONResponse:
    return JSONResponse({"tools": TOOLS})


@app.post("/tools/get_profile")
async def get_profile(body: dict) -> JSONResponse:
    customer_id = body["customer_id"]
    async with get_db_session() as session:
        c = (await session.execute(select(Customer).where(Customer.id == customer_id))).scalar_one()
    return JSONResponse({"id": c.id, "name": c.name, "segment": c.segment})


@app.post("/tools/get_accounts")
async def get_accounts(body: dict) -> JSONResponse:
    customer_id = body["customer_id"]
    async with get_db_session() as session:
        rows = (
            await session.execute(select(Account).where(Account.customer_id == customer_id))
        ).scalars().all()
    return JSONResponse([
        {"id": a.id, "kind": a.kind, "balance_nok": str(a.balance_nok)} for a in rows
    ])


@app.post("/tools/get_transactions")
async def get_transactions(body: dict) -> JSONResponse:
    customer_id = body["customer_id"]
    days = body.get("days", 90)
    async with get_db_session() as session:
        accounts = (
            await session.execute(select(Account).where(Account.customer_id == customer_id))
        ).scalars().all()
        acc_ids = [a.id for a in accounts]
        rows = (
            await session.execute(
                select(Transaction).where(Transaction.account_id.in_(acc_ids)).limit(200)
            )
        ).scalars().all()
    return JSONResponse([
        {"id": t.id, "account_id": t.account_id, "amount_nok": str(t.amount_nok),
         "merchant": t.merchant, "category": t.category, "is_recurring": t.is_recurring}
        for t in rows
    ])


@app.post("/tools/get_holdings")
async def get_holdings(body: dict) -> JSONResponse:
    customer_id = body["customer_id"]
    async with get_db_session() as session:
        rows = (
            await session.execute(select(Holding).where(Holding.customer_id == customer_id))
        ).scalars().all()
    return JSONResponse([
        {"id": h.id, "ticker": h.ticker, "quantity": str(h.quantity),
         "avg_cost_nok": str(h.avg_cost_nok)} for h in rows
    ])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7001)
