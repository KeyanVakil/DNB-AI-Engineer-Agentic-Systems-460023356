"""LangGraph StateGraph definition for FinSight multi-agent orchestration."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

_log = logging.getLogger(__name__)


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
            # Give the event loop a chance to process other pending work
            # (e.g. a concurrent fault-injection request from the test harness)
            # before we execute this node. Without this yield, a graph built
            # from in-memory fakes can complete in a single event-loop turn
            # and starve sibling tasks.
            await asyncio.sleep(0)
            # Test-only fault injection: check `_fault_slots` right before each
            # node runs. This closes the race where POST /runs returns and the
            # background graph starts before the test's inject_fault request
            # arrives — by polling here, a late-arriving fault still lands on
            # the named node.
            if self._thread_id:
                from app.api.runs import _fault_slots

                fault = _fault_slots.get(self._thread_id)
                if fault and fault[0] == name:
                    _fault_slots.pop(self._thread_id, None)
                    raise RuntimeError(fault[1])
            node = self.nodes[name]
            update = await node(state, llm=self._llm, mcp=self._mcp)
            # Honor the GraphState `errors` reducer: accumulate, don't overwrite.
            # Otherwise concurrent specialists race and only the last node's
            # errors survive — losing visibility into MCP tool failures.
            new_errors = update.pop("errors", None)
            state.update(update)
            if new_errors:
                existing = state.get("errors") or []
                state["errors"] = list(existing) + list(new_errors)
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
            # Checkpoint table may not exist yet on first boot, or the DB
            # may be transiently unavailable. A missing checkpoint is not
            # fatal — we just restart from scratch — but surface the cause
            # so operators see it.
            _log.warning(
                "load_checkpoint failed for thread_id=%s", self._thread_id, exc_info=True
            )
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
            # Save failures mean resume won't work for this run, but they
            # shouldn't crash the in-flight graph. Log so the operator can
            # diagnose; common cause is a misconfigured DATABASE_URL.
            _log.warning(
                "save_checkpoint failed for thread_id=%s", self._thread_id, exc_info=True
            )


def build_graph(
    llm: Any,
    mcp: dict[str, Any],
    thread_id: str | None = None,
) -> _CompiledGraph:
    """Build and return the FinSight compiled graph."""
    from app.agents import (
        compliance,
        coordinator,
        market,
        portfolio,
        transactions,
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
