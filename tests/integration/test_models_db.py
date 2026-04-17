"""Data model tests: CRUD, relationships, and constraints.

Covers every table from PRD section 5:
- customers / accounts / transactions / holdings
- prices / news
- rules
- runs / eval_results / human_reviews / drift_events
- memory_embeddings (pgvector)
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError, IntegrityError

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Customer side
# ---------------------------------------------------------------------------


async def test_customer_account_transaction_cascade(db_session, seed_fixtures):
    from app.memory.models import Account, Customer, Transaction

    customer = (
        await db_session.execute(select(Customer).where(Customer.id == "c-0001"))
    ).scalar_one()
    assert customer.segment == "mass_affluent"

    accounts = (
        await db_session.execute(select(Account).where(Account.customer_id == "c-0001"))
    ).scalars().all()
    assert len(accounts) == 3
    assert {a.kind for a in accounts} == {"checking", "savings", "credit"}

    txns = (
        await db_session.execute(
            select(Transaction).where(Transaction.account_id == "a-100")
        )
    ).scalars().all()
    assert len(txns) >= 3
    assert all(isinstance(t.amount_nok, Decimal) for t in txns)


async def test_account_balance_precision_matches_schema(db_session):
    """NUMERIC(14,2) — losing a decimal would be a silent correctness bug."""
    from app.memory.models import Account

    row = (await db_session.execute(select(Account).where(Account.id == "a-100"))).scalar_one()
    assert row.balance_nok == Decimal("42310.55")


async def test_account_kind_enum_rejects_invalid_value(db_session):
    from app.memory.models import Account

    db_session.add(
        Account(
            id="a-bad",
            customer_id="c-0001",
            kind="martian",  # not in enum
            balance_nok=Decimal("1.00"),
            opened_at=dt.datetime.now(dt.UTC),
        )
    )
    with pytest.raises(DBAPIError):
        await db_session.flush()


async def test_transactions_is_recurring_defaults_to_false(db_session):
    from app.memory.models import Transaction

    t = Transaction(
        id="t-new",
        account_id="a-100",
        ts=dt.datetime.now(dt.UTC),
        amount_nok=Decimal("-10.00"),
        merchant="Test",
        category="misc",
    )
    db_session.add(t)
    await db_session.flush()
    assert t.is_recurring is False


async def test_holdings_foreign_key_enforced(db_session):
    from app.memory.models import Holding

    db_session.add(
        Holding(
            id="h-bad",
            customer_id="c-does-not-exist",
            ticker="FAKE.OL",
            quantity=Decimal("1"),
            avg_cost_nok=Decimal("1.00"),
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


# ---------------------------------------------------------------------------
# Market side
# ---------------------------------------------------------------------------


async def test_prices_primary_key_is_ticker_plus_date(db_session):
    from app.memory.models import Price

    row = Price(
        ticker="EQNR.OL",
        ts=dt.date(2026, 4, 14),  # already seeded -> collision
        close=Decimal("999.99"),
    )
    db_session.add(row)
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_news_sentiment_is_numeric(db_session):
    from app.memory.models import News

    rows = (await db_session.execute(select(News).where(News.ticker == "EQNR.OL"))).scalars().all()
    assert rows
    assert all(-1 <= float(n.sentiment) <= 1 for n in rows)


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------


async def test_rule_code_is_unique(db_session):
    from app.memory.models import Rule

    db_session.add(
        Rule(
            id="r-dup",
            code="NO_PII",  # already seeded
            description="dup",
            severity="warn",
            regex_pattern=None,
            applies_to="report",
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_rule_applies_to_enum(db_session):
    from app.memory.models import Rule

    db_session.add(
        Rule(
            id="r-bad",
            code="X",
            description="x",
            severity="info",
            regex_pattern=None,
            applies_to="martian",
        )
    )
    with pytest.raises(DBAPIError):
        await db_session.flush()


# ---------------------------------------------------------------------------
# Application tables
# ---------------------------------------------------------------------------


async def test_run_status_transitions(db_session, customer_id):
    from app.memory.models import Run

    run = Run(
        id=uuid.uuid4(),
        customer_id=customer_id,
        status="queued",
        prompt_version="v1",
        model_config={"id": "llama3.1-8b"},
        started_at=None,
    )
    db_session.add(run)
    await db_session.flush()

    run.status = "running"
    run.started_at = dt.datetime.now(dt.UTC)
    await db_session.flush()

    run.status = "succeeded"
    run.finished_at = dt.datetime.now(dt.UTC)
    run.total_tokens = 800
    run.total_cost_usd = Decimal("0.0123")
    run.report_md = "# ok"
    run.report_json = {"recommendations": ["a", "b", "c"]}
    await db_session.flush()

    fetched = (
        await db_session.execute(select(Run).where(Run.id == run.id))
    ).scalar_one()
    assert fetched.status == "succeeded"
    assert fetched.total_cost_usd == Decimal("0.0123")


async def test_run_status_enum_rejects_unknown_value(db_session, customer_id):
    from app.memory.models import Run

    db_session.add(
        Run(
            id=uuid.uuid4(),
            customer_id=customer_id,
            status="martian",
            prompt_version="v1",
            model_config={},
        )
    )
    with pytest.raises(DBAPIError):
        await db_session.flush()


async def test_eval_result_kind_enum(db_session, customer_id):
    from app.memory.models import EvalResult, Run

    run = Run(
        id=uuid.uuid4(), customer_id=customer_id, status="queued",
        prompt_version="v1", model_config={},
    )
    db_session.add(run)
    await db_session.flush()

    for kind in ("code", "judge", "human"):
        db_session.add(
            EvalResult(
                id=uuid.uuid4(),
                run_id=run.id,
                kind=kind,
                score=Decimal("4.0"),
                passed=True,
                payload={},
            )
        )
    await db_session.flush()


async def test_human_review_score_range_enforced_by_db_or_model(db_session, customer_id):
    """Score must be an integer in 1-5. Enforced either via CHECK constraint or
    via model validation — either is acceptable, but out-of-range must fail."""
    from app.memory.models import HumanReview, Run

    run = Run(
        id=uuid.uuid4(), customer_id=customer_id, status="succeeded",
        prompt_version="v1", model_config={},
    )
    db_session.add(run)
    await db_session.flush()

    db_session.add(
        HumanReview(
            id=uuid.uuid4(),
            run_id=run.id,
            reviewer="x",
            score=99,
            approved=True,
            notes=None,
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_drift_event_requires_window_range(db_session):
    from app.memory.models import DriftEvent

    now = dt.datetime.now(dt.UTC)
    event = DriftEvent(
        id=uuid.uuid4(),
        metric="judge_score",
        baseline_window=(now - dt.timedelta(hours=24), now - dt.timedelta(hours=1)),
        current_window=(now - dt.timedelta(hours=1), now),
        statistic=Decimal("0.35"),
        p_value=Decimal("0.002"),
        psi=Decimal("0.28"),
        severity="alert",
    )
    db_session.add(event)
    await db_session.flush()
    assert event.id is not None


# ---------------------------------------------------------------------------
# Embeddings (pgvector)
# ---------------------------------------------------------------------------


async def test_memory_embeddings_roundtrip_with_pgvector(db_session, customer_id):
    from app.memory.embeddings import nearest_neighbors, store_embedding
    from app.memory.models import Run

    run = Run(
        id=uuid.uuid4(), customer_id=customer_id, status="succeeded",
        prompt_version="v1", model_config={},
    )
    db_session.add(run)
    await db_session.flush()

    vec = [0.1] * 768
    await store_embedding(run_id=run.id, role="assistant", text="hello", embedding=vec)

    neighbors = await nearest_neighbors(vec, k=1)
    assert neighbors
    assert neighbors[0].run_id == run.id
    assert neighbors[0].text == "hello"


async def test_memory_embeddings_rejects_wrong_dimension(db_session, customer_id):
    from app.memory.embeddings import store_embedding
    from app.memory.models import Run

    run = Run(
        id=uuid.uuid4(), customer_id=customer_id, status="succeeded",
        prompt_version="v1", model_config={},
    )
    db_session.add(run)
    await db_session.flush()

    with pytest.raises(ValueError):
        await store_embedding(
            run_id=run.id, role="assistant", text="wrong dim", embedding=[0.1] * 10
        )
