"""Seed fixtures into the database for testing and local dev."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.memory.models import (
    Account,
    Customer,
    DriftEvent,
    EvalResult,
    Holding,
    News,
    Price,
    Rule,
    Transaction,
)


async def load_customers(session: AsyncSession, customers: list[dict[str, Any]]) -> None:
    for c in customers:
        profile = c.get("profile", c)
        customer = Customer(
            id=profile["id"],
            name=profile["name"],
            segment=profile["segment"],
            joined_at=dt.datetime.fromisoformat(profile["joined_at"].replace("Z", "+00:00")),
        )
        session.add(customer)

        for acc in c.get("accounts", []):
            session.add(
                Account(
                    id=acc["id"],
                    customer_id=acc["customer_id"],
                    kind=acc["kind"],
                    balance_nok=Decimal(str(acc["balance_nok"])),
                    opened_at=dt.datetime.fromisoformat(acc["opened_at"].replace("Z", "+00:00")),
                )
            )

        for txn in c.get("transactions", []):
            session.add(
                Transaction(
                    id=txn["id"],
                    account_id=txn["account_id"],
                    ts=dt.datetime.fromisoformat(txn["ts"].replace("Z", "+00:00")),
                    amount_nok=Decimal(str(txn["amount_nok"])),
                    merchant=txn["merchant"],
                    category=txn["category"],
                    is_recurring=txn.get("is_recurring", False),
                )
            )

        for h in c.get("holdings", []):
            session.add(
                Holding(
                    id=h["id"],
                    customer_id=h["customer_id"],
                    ticker=h["ticker"],
                    quantity=Decimal(str(h["quantity"])),
                    avg_cost_nok=Decimal(str(h["avg_cost_nok"])),
                )
            )


async def load_market(session: AsyncSession, market: dict[str, Any]) -> None:
    for ticker, rows in market.get("prices", {}).items():
        for row in rows:
            session.add(
                Price(
                    ticker=row["ticker"],
                    ts=dt.date.fromisoformat(row["ts"]),
                    close=Decimal(str(row["close"])),
                )
            )

    for item in market.get("news", {}).values():
        for n in item:
            session.add(
                News(
                    id=n["id"],
                    ts=dt.datetime.fromisoformat(n["ts"].replace("Z", "+00:00")),
                    ticker=n["ticker"],
                    headline=n["headline"],
                    sentiment=Decimal(str(n["sentiment"])),
                )
            )


async def load_compliance(session: AsyncSession, rules: list[dict[str, Any]]) -> None:
    for r in rules:
        session.add(
            Rule(
                id=r["id"],
                code=r["code"],
                description=r["description"],
                severity=r["severity"],
                regex_pattern=r.get("regex_pattern"),
                applies_to=r["applies_to"],
            )
        )


async def load_eval_results(
    session: AsyncSession, records: list[dict[str, Any]]
) -> None:
    import uuid

    for r in records:
        session.add(
            EvalResult(
                id=uuid.uuid4(),
                run_id=r.get("run_id"),
                kind=r["kind"],
                score=Decimal(str(r["score"])),
                passed=r.get("passed", True),
                payload=r.get("payload", {}),
                created_at=r.get("created_at", dt.datetime.now(dt.timezone.utc)),
            )
        )
