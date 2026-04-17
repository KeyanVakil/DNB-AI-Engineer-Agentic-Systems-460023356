"""pgvector embedding storage and retrieval."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.memory.models import MemoryEmbedding

_EXPECTED_DIM = 768

# Module-level session override for use in tests (set by dependency injection)
_session_override: AsyncSession | None = None


async def store_embedding(
    *,
    run_id: uuid.UUID,
    role: str,
    text: str,
    embedding: list[float],
) -> MemoryEmbedding:
    if len(embedding) != _EXPECTED_DIM:
        raise ValueError(f"Embedding dimension {len(embedding)} != expected {_EXPECTED_DIM}")

    record = MemoryEmbedding(
        id=uuid.uuid4(),
        run_id=run_id,
        role=role,
        text=text,
        embedding=embedding,
    )
    from app.memory.database import get_db_session

    async with get_db_session() as session:
        session.add(record)
    return record


async def nearest_neighbors(
    vector: list[float], k: int = 5
) -> list[MemoryEmbedding]:
    if len(vector) != _EXPECTED_DIM:
        raise ValueError(f"Query vector dimension {len(vector)} != expected {_EXPECTED_DIM}")

    from app.memory.database import get_db_session

    async with get_db_session() as session:
        result = await session.execute(
            select(MemoryEmbedding)
            .order_by(MemoryEmbedding.embedding.l2_distance(vector))  # type: ignore[attr-defined]
            .limit(k)
        )
        return list(result.scalars().all())
