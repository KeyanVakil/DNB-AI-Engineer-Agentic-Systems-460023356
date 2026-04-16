from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TSTZRANGE, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


AccountKind = Enum("checking", "savings", "credit", name="account_kind")
RunStatus = Enum("queued", "running", "succeeded", "failed", name="run_status")
EvalKind = Enum("code", "judge", "human", name="eval_kind")
Severity = Enum("info", "warn", "alert", name="severity")
AppliesTo = Enum("report", "recommendation", name="applies_to")


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    segment: Mapped[str] = mapped_column(String, nullable=False)
    joined_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    accounts: Mapped[list[Account]] = relationship("Account", back_populates="customer")
    holdings: Mapped[list[Holding]] = relationship("Holding", back_populates="customer")


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    customer_id: Mapped[str] = mapped_column(String, ForeignKey("customers.id"), nullable=False)
    kind: Mapped[str] = mapped_column(AccountKind, nullable=False)
    balance_nok: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    opened_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    customer: Mapped[Customer] = relationship("Customer", back_populates="accounts")
    transactions: Mapped[list[Transaction]] = relationship("Transaction", back_populates="account")


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    account_id: Mapped[str] = mapped_column(String, ForeignKey("accounts.id"), nullable=False)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    amount_nok: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    merchant: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    is_recurring: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    account: Mapped[Account] = relationship("Account", back_populates="transactions")


class Holding(Base):
    __tablename__ = "holdings"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    customer_id: Mapped[str] = mapped_column(String, ForeignKey("customers.id"), nullable=False)
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    avg_cost_nok: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)

    customer: Mapped[Customer] = relationship("Customer", back_populates="holdings")


class Price(Base):
    __tablename__ = "prices"
    __table_args__ = (UniqueConstraint("ticker", "ts", name="uq_prices_ticker_ts"),)

    ticker: Mapped[str] = mapped_column(String, primary_key=True)
    ts: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    close: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)


class News(Base):
    __tablename__ = "news"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    sentiment: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)


class Rule(Base):
    __tablename__ = "rules"
    __table_args__ = (UniqueConstraint("code", name="uq_rules_code"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    code: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(Severity, nullable=False)
    regex_pattern: Mapped[str | None] = mapped_column(Text, nullable=True)
    applies_to: Mapped[str] = mapped_column(AppliesTo, nullable=False)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id: Mapped[str] = mapped_column(String, ForeignKey("customers.id"), nullable=False)
    status: Mapped[str] = mapped_column(RunStatus, nullable=False, default="queued")
    prompt_version: Mapped[str] = mapped_column(String, nullable=False, default="v1")
    model_config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    report_md: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    thread_id: Mapped[str | None] = mapped_column(String, nullable=True)

    eval_results: Mapped[list[EvalResult]] = relationship("EvalResult", back_populates="run")
    human_reviews: Mapped[list[HumanReview]] = relationship("HumanReview", back_populates="run")


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id"), nullable=True
    )
    kind: Mapped[str] = mapped_column(EvalKind, nullable=False)
    score: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.timezone.utc)
    )

    run: Mapped[Run | None] = relationship("Run", back_populates="eval_results")


class HumanReview(Base):
    __tablename__ = "human_reviews"
    __table_args__ = (CheckConstraint("score >= 1 AND score <= 5", name="ck_human_reviews_score"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False
    )
    reviewer: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    approved: Mapped[bool] = mapped_column(Boolean, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.timezone.utc)
    )

    run: Mapped[Run] = relationship("Run", back_populates="human_reviews")


class DriftEvent(Base):
    __tablename__ = "drift_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric: Mapped[str] = mapped_column(String, nullable=False)
    baseline_window: Mapped[tuple[dt.datetime, dt.datetime]] = mapped_column(
        TSTZRANGE, nullable=False
    )
    current_window: Mapped[tuple[dt.datetime, dt.datetime]] = mapped_column(
        TSTZRANGE, nullable=False
    )
    statistic: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    p_value: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    psi: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    severity: Mapped[str] = mapped_column(Severity, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class MemoryEmbedding(Base):
    __tablename__ = "memory_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # No FK — embeddings may be stored across connection boundaries in tests
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Any] = mapped_column(Vector(768), nullable=False)
