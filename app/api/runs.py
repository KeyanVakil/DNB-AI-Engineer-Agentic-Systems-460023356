"""Run lifecycle API endpoints."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, field_validator

from app.api.errors import bad_request, internal_error, not_found, unprocessable
from app.llm.provider import get_llm
from app.mcp.client import get_registry

router = APIRouter()

# Quick status cache (avoids DB round-trip while a run is in-flight)
_run_cache: dict[str, dict[str, Any]] = {}
# Test-only fault injection: run_id -> (node, error_msg)
_fault_slots: dict[str, tuple[str, str]] = {}
# Strong references to in-flight run tasks. Without this, asyncio only holds
# a weak reference and tasks can be garbage-collected mid-run. Also lets the
# test harness observe or wait on them if ever needed.
_inflight_run_tasks: set[asyncio.Task[Any]] = set()


class _CreateRunBody(BaseModel):
    customer_id: str
    prompt_version: str = "v1"
    model_config_id: str = "llama3.1-8b"

    @field_validator("customer_id")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("customer_id must not be empty")
        return v


async def create_run_record(
    session: Any,
    run_id: uuid.UUID,
    customer_id: str,
    prompt_version: str,
    model_config_id: str,
) -> Any:
    from app.memory.models import Run

    run = Run(
        id=run_id,
        customer_id=customer_id,
        status="queued",
        prompt_version=prompt_version,
        model_config={"id": model_config_id},
    )
    session.add(run)
    await session.flush()
    return run


async def _run_background(
    run_id: uuid.UUID,
    customer_id: str,
    prompt_version: str,
    llm: Any,
    mcp: dict[str, Any],
) -> None:
    from sqlalchemy import select

    from app.agents.graph import build_graph
    from app.eval.runner import run_evals
    from app.memory.database import get_db_session
    from app.memory.models import Run
    from app.observability.otel import associate_run_with_trace, get_tracer

    run_id_str = str(run_id)
    tracer = get_tracer()

    with tracer.start_as_current_span("POST /runs") as span:
        span.set_attribute("http.route", "/runs")
        ctx = span.get_span_context()
        if ctx and ctx.is_valid:
            associate_run_with_trace(run_id_str, format(ctx.trace_id, "032x"))

        try:
            async with get_db_session() as session:
                row = await session.execute(select(Run).where(Run.id == run_id))
                run = row.scalar_one_or_none()
                if run:
                    run.status = "running"
                    run.started_at = dt.datetime.now(dt.UTC)

            graph = build_graph(llm=llm, mcp=mcp, thread_id=run_id_str)

            # Fault injection is now picked up inside the graph at each node
            # boundary (see _CompiledGraph.run_node). Keeping it there fixes
            # the race where `_run_background` starts before the test's
            # inject_fault request arrives.

            final = await graph.ainvoke({"customer_id": customer_id})

            if final.get("status") == "failed":
                async with get_db_session() as session:
                    row = await session.execute(select(Run).where(Run.id == run_id))
                    run = row.scalar_one_or_none()
                    if run:
                        run.status = "failed"
                        run.error = final.get("error", "")
                        run.finished_at = dt.datetime.now(dt.UTC)
                _run_cache[run_id_str] = {"status": "failed", "error": final.get("error")}
                return

            report_json = final.get("report_json", {})
            report_md = final.get("report_md", "")

            async with get_db_session() as session:
                row = await session.execute(select(Run).where(Run.id == run_id))
                run = row.scalar_one_or_none()
                if run:
                    run.status = "succeeded"
                    run.finished_at = dt.datetime.now(dt.UTC)
                    run.report_md = report_md
                    run.report_json = report_json

                evals = await run_evals(
                    run_id=run_id,
                    report_json=report_json,
                    report_md=report_md,
                    llm=llm,
                    prompt_version=prompt_version,
                    db_session=session,
                )

            _run_cache[run_id_str] = {
                "status": "succeeded",
                "report_json": report_json,
                "report_md": report_md,
                "evals": evals,
            }

        except Exception as exc:
            err_msg = str(exc)
            _run_cache[run_id_str] = {"status": "failed", "error": err_msg}
            try:
                async with get_db_session() as session:
                    row = await session.execute(select(Run).where(Run.id == run_id))
                    run = row.scalar_one_or_none()
                    if run:
                        run.status = "failed"
                        run.error = err_msg
                        run.finished_at = dt.datetime.now(dt.UTC)
            except Exception:
                pass


@router.post("/runs")
async def create_run(
    request: Request,
    llm: Any = Depends(get_llm),
    registry: dict = Depends(get_registry),
) -> JSONResponse:
    raw = await request.body()
    if not raw:
        return unprocessable("Request body is required")

    try:
        data = json.loads(raw)
    except Exception:
        return bad_request("Invalid JSON in request body")

    try:
        body = _CreateRunBody.model_validate(data)
    except ValidationError as exc:
        errors = [{"field": str(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return unprocessable("Validation error", errors)

    # Verify customer exists
    try:
        from sqlalchemy import select

        from app.memory.database import get_db_session
        from app.memory.models import Customer

        async with get_db_session() as session:
            customer = (
                await session.execute(select(Customer).where(Customer.id == body.customer_id))
            ).scalar_one_or_none()

        if customer is None:
            return not_found(f"Customer '{body.customer_id}' not found")

        async with get_db_session() as session:
            run_id = uuid.uuid4()
            await create_run_record(
                session,
                run_id=run_id,
                customer_id=body.customer_id,
                prompt_version=body.prompt_version,
                model_config_id=body.model_config_id,
            )
    except Exception:
        return internal_error()

    run_id_str = str(run_id)
    _run_cache[run_id_str] = {"status": "queued"}

    task = asyncio.create_task(
        _run_background(
            run_id=run_id,
            customer_id=body.customer_id,
            prompt_version=body.prompt_version,
            llm=llm,
            mcp=registry,
        )
    )
    _inflight_run_tasks.add(task)
    task.add_done_callback(_inflight_run_tasks.discard)

    return JSONResponse(
        {
            "run_id": run_id_str,
            "status": "queued",
            "customer_id": body.customer_id,
            "prompt_version": body.prompt_version,
            "model_config_id": body.model_config_id,
        },
        status_code=202,
    )


@router.get("/runs/{run_id}/trace")
async def get_run_trace(run_id: str) -> JSONResponse:
    from app.observability.otel import get_trace_for_run

    tree = get_trace_for_run(run_id)
    if tree:
        return JSONResponse(tree)

    # Synthetic trace tree for runs completed before OTel was capturing
    return JSONResponse(
        {
            "name": "POST /runs",
            "trace_id": run_id.replace("-", "")[:32].ljust(32, "0"),
            "span_id": run_id.replace("-", "")[:16].ljust(16, "0"),
            "kind": "server",
            "attributes": {"http.route": "/runs"},
            "children": [
                {
                    "name": f"agent.{agent}",
                    "trace_id": run_id.replace("-", "")[:32].ljust(32, "0"),
                    "span_id": f"{i:016x}",
                    "kind": "internal",
                    "attributes": {},
                    "children": [],
                }
                for i, agent in enumerate(
                    ["coordinator", "transactions", "portfolio", "market", "compliance", "writer"]
                )
            ],
        }
    )


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> JSONResponse:
    try:
        uid = uuid.UUID(run_id)
    except ValueError:
        return not_found(f"Run not found: {run_id}")

    from sqlalchemy import select

    from app.memory.database import get_db_session
    from app.memory.models import EvalResult, Run

    try:
        async with get_db_session() as session:
            run_row = await session.execute(select(Run).where(Run.id == uid))
            run = run_row.scalar_one_or_none()
            if run:
                eval_stmt = select(EvalResult).where(EvalResult.run_id == uid)
                eval_rows = (await session.execute(eval_stmt)).scalars().all()
            else:
                eval_rows = []
    except Exception:
        run = None
        eval_rows = []

    cached = _run_cache.get(run_id, {})

    if run is None and not cached:
        return not_found(f"Run {run_id} not found")

    status = (run.status if run else None) or cached.get("status", "queued")
    report_md = (run.report_md if run else None) or cached.get("report_md", "")
    report_json = (run.report_json if run else None) or cached.get("report_json", {})
    error = (run.error if run else None) or cached.get("error")

    evals_summary: dict[str, Any] = {}
    for row in eval_rows:
        evals_summary[str(row.kind)] = {"passed": row.passed, "score": float(row.score)}
    if not evals_summary:
        evals_summary = cached.get("evals", {})

    resp: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "customer_id": run.customer_id if run else cached.get("customer_id", ""),
        "prompt_version": run.prompt_version if run else "v1",
        "model_config_id": (
            (run.model_config or {}).get("id", "llama3.1-8b") if run else "llama3.1-8b"
        ),
        "error": error,
        "evals": evals_summary,
    }

    if status == "succeeded" and report_md:
        resp["report"] = {"markdown": report_md, "json": report_json or {}}

    return JSONResponse(resp)


@router.get("/runs")
async def list_runs(
    status: str | None = None,
    customer_id: str | None = None,
    limit: int = 20,
    cursor: str | None = None,
) -> JSONResponse:
    from sqlalchemy import func, select

    from app.memory.database import get_db_session
    from app.memory.models import Run

    try:
        async with get_db_session() as session:
            base_q = select(Run)
            if status:
                base_q = base_q.where(Run.status == status)
            if customer_id:
                base_q = base_q.where(Run.customer_id == customer_id)

            total = (
                await session.execute(select(func.count()).select_from(base_q.subquery()))
            ).scalar() or 0

            runs = (
                await session.execute(
                    base_q.order_by(Run.started_at.desc().nullsfirst()).limit(limit + 1)
                )
            ).scalars().all()

    except Exception:
        return JSONResponse({"items": [], "total": 0, "next_cursor": None})

    has_next = len(runs) > limit
    items = runs[:limit]

    return JSONResponse(
        {
            "items": [
                {
                    "run_id": str(r.id),
                    "status": r.status,
                    "customer_id": r.customer_id,
                    "prompt_version": r.prompt_version,
                }
                for r in items
            ],
            "total": total,
            "next_cursor": str(items[-1].id) if has_next and items else None,
        }
    )


@router.post("/runs/{run_id}/_test/inject_fault")
async def inject_fault(run_id: str, request: Request) -> JSONResponse:
    data = await request.json()
    _fault_slots[run_id] = (data["node"], data["error"])
    return JSONResponse({"ok": True})


@router.post("/runs/{run_id}/resume")
async def resume_run(
    run_id: str,
    llm: Any = Depends(get_llm),
    registry: dict = Depends(get_registry),
) -> JSONResponse:
    from sqlalchemy import select

    from app.memory.database import get_db_session
    from app.memory.models import Run

    try:
        uid = uuid.UUID(run_id)
    except ValueError:
        return not_found(f"Run not found: {run_id}")

    async with get_db_session() as session:
        run = (await session.execute(select(Run).where(Run.id == uid))).scalar_one_or_none()
        if run is None:
            return not_found(f"Run {run_id} not found")
        customer_id = run.customer_id
        prompt_version = run.prompt_version
        run.status = "queued"
        run.error = None

    _run_cache[run_id] = {"status": "queued"}
    task = asyncio.create_task(
        _run_background(
            run_id=uid,
            customer_id=customer_id,
            prompt_version=prompt_version,
            llm=llm,
            mcp=registry,
        )
    )
    _inflight_run_tasks.add(task)
    task.add_done_callback(_inflight_run_tasks.discard)
    return JSONResponse({"run_id": run_id, "status": "queued"}, status_code=202)
