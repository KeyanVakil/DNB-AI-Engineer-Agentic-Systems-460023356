"""LangGraph StateGraph definition for FinSight multi-agent orchestration."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from app.agents import state as state_module


class _Node:
    """Wraps a node function and supports fault injection for testing."""

    def __init__(self, name: str, fn: Callable[..., Any]) -> None:
        self.name = name
        self._fn = fn
        self._fault: Exception | None = None

    def inject_fault(self, exc: Exception) -> None:
        self._fault = exc

    async def __call__(self, state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if self._fault is not None:
            err = self._fault
            self._fault = None
            raise err
        return await self._fn(state, **kwargs)


class _CompiledGraph:
    """Minimal async graph executor that mirrors the LangGraph compiled graph API."""

    def __init__(
        self,
        nodes: dict[str, _Node],
        edges: list[tuple[str, str]],
        conditional_edges: dict[str, set[str]],
        llm: Any,
        mcp: dict[str, Any],
        thread_id: str | None = None,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self._llm = llm
        self._mcp = mcp
        self._thread_id = thread_id

    async def ainvoke(self, input_state: dict[str, Any]) -> dict[str, Any]:
        state: dict[str, Any] = {
            "errors": [],
            "revision_count": 0,
            "status": "running",
            "error": None,
            **input_state,
        }

        checkpoint = await self._load_checkpoint(state)
        completed = set(checkpoint.get("completed_nodes", []))
        state.update(checkpoint.get("state", {}))
        state["customer_id"] = input_state["customer_id"]

        async def run_node(name: str) -> None:
            if name in completed:
                return
            node = self.nodes[name]
            update = await node(state, llm=self._llm, mcp=self._mcp)
            state.update(update)
            completed.add(name)
            await self._save_checkpoint(state, completed)

        # 1. Coordinator
        await run_node("coordinator")

        # 2. Specialists run concurrently
        if not all(n in completed for n in ("transactions", "portfolio", "market")):
            await asyncio.gather(
                run_node("transactions"),
                run_node("portfolio"),
                run_node("market"),
            )

        # 3. Compliance → writer loop (max 2 revisions)
        MAX_REVISIONS = 2
        while True:
            if "compliance" not in completed:
                await run_node("compliance")

            compliance_result = state.get("compliance_result", {})
            decision = compliance_result.get("decision", "approve")

            if decision == "approve":
                break

            # Revise: re-run writer then loop back to compliance
            revision_count = state.get("revision_count", 0)
            if revision_count >= MAX_REVISIONS:
                state["status"] = "failed"
                state["error"] = (
                    "Compliance rejected the report after maximum revisions. "
                    "Please review compliance guidelines."
                )
                return state

            state["revision_count"] = revision_count + 1
            completed.discard("compliance")
            completed.discard("writer")
            await run_node("writer")

        # 4. Writer (final)
        await run_node("writer")

        return state

    async def _load_checkpoint(self, state: dict[str, Any]) -> dict[str, Any]:
        if not self._thread_id:
            return {}
        try:
            from sqlalchemy import text
            from app.memory.database import get_db_session

            async with get_db_session() as session:
                row = await session.execute(
                    text(
                        "SELECT checkpoint FROM langgraph.checkpoints "
                        "WHERE thread_id = :tid ORDER BY checkpoint_id DESC LIMIT 1"
                    ),
                    {"tid": self._thread_id},
                )
                record = row.fetchone()
                if record and record[0]:
                    import json

                    return json.loads(record[0]) if isinstance(record[0], str) else record[0]
        except Exception:
            pass
        return {}

    async def _save_checkpoint(self, state: dict[str, Any], completed: set[str]) -> None:
        if not self._thread_id:
            return
        try:
            import json
            import uuid
            from sqlalchemy import text
            from app.memory.database import get_db_session

            # Store a JSON-safe snapshot (exclude non-serialisable items)
            snapshot: dict[str, Any] = {
                "state": {
                    k: v
                    for k, v in state.items()
                    if k
                    not in ("errors",)
                },
                "completed_nodes": list(completed),
            }
            async with get_db_session() as session:
                await session.execute(
                    text(
                        "INSERT INTO langgraph.checkpoints "
                        "(thread_id, checkpoint_ns, checkpoint_id, checkpoint) "
                        "VALUES (:tid, '', :cid, :data) "
                        "ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) DO NOTHING"
                    ),
                    {
                        "tid": self._thread_id,
                        "cid": str(uuid.uuid4()),
                        "data": json.dumps(snapshot),
                    },
                )
        except Exception:
            pass


def build_graph(
    llm: Any,
    mcp: dict[str, Any],
    thread_id: str | None = None,
) -> _CompiledGraph:
    """Build and return the FinSight compiled graph."""
    from app.agents import (
        coordinator,
        transactions,
        portfolio,
        market,
        compliance,
        writer,
    )

    nodes = {
        "coordinator": _Node("coordinator", coordinator.run),
        "transactions": _Node("transactions", transactions.run),
        "portfolio": _Node("portfolio", portfolio.run),
        "market": _Node("market", market.run),
        "compliance": _Node("compliance", compliance.run),
        "writer": _Node("writer", writer.run),
    }

    # Fixed edges (unconditional sequencing in the happy path)
    edges: list[tuple[str, str]] = [
        ("coordinator", "transactions"),
        ("coordinator", "portfolio"),
        ("coordinator", "market"),
        ("transactions", "compliance"),
        ("portfolio", "compliance"),
        ("market", "compliance"),
        ("compliance", "writer"),
        ("writer", "__end__"),
    ]

    # Conditional edges: compliance can route back to specialists or writer
    conditional_edges: dict[str, set[str]] = {
        "compliance": {"writer", "transactions", "portfolio"},
    }

    return _CompiledGraph(
        nodes=nodes,
        edges=edges,
        conditional_edges=conditional_edges,
        llm=llm,
        mcp=mcp,
        thread_id=thread_id,
    )
