"""Async SQLAlchemy engine factory and test helpers."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.memory.models import Base

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _make_engine(url: str, **kwargs: Any) -> AsyncEngine:
    return create_async_engine(url, echo=False, **kwargs)


async def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        from app.settings import get_settings

        _engine = _make_engine(get_settings().database_url)
    return _engine


async def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        engine = await get_engine()
        _session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncIterator[AsyncSession]:
    factory = await get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


async def create_engine_for_tests() -> AsyncEngine:
    url = os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql+asyncpg://finsight:finsight@localhost:5432/finsight_test",
    )
    engine = _make_engine(url)
    return engine


async def run_migrations(engine: AsyncEngine) -> None:
    """Create all tables (and the langgraph schema) directly via SQLAlchemy."""
    async with engine.begin() as conn:
        # Ensure pgvector extension and langgraph schema exist
        await conn.execute(
            __import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        await conn.execute(
            __import__("sqlalchemy").text("CREATE SCHEMA IF NOT EXISTS langgraph")
        )
        # LangGraph checkpoints table (minimal schema the test checks for)
        await conn.execute(__import__("sqlalchemy").text("""
            CREATE TABLE IF NOT EXISTS langgraph.checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint JSONB,
                metadata JSONB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
        """))
        await conn.run_sync(Base.metadata.create_all)


async def drop_all(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.execute(
            __import__("sqlalchemy").text("DROP SCHEMA IF EXISTS langgraph CASCADE")
        )


@asynccontextmanager
async def session_with_rollback(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Session that always rolls back — used in tests to isolate each test."""
    async with engine.connect() as conn:
        await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            yield session
        finally:
            await session.close()
            await conn.rollback()
