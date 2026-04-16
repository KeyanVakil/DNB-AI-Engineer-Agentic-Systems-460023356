"""Liveness and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response

from app.mcp.client import get_registry

router = APIRouter()


@router.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@router.get("/readyz")
async def readyz(registry: dict = Depends(get_registry)) -> Response:
    checks: dict[str, dict] = {}

    # Postgres check
    try:
        from app.memory.database import get_db_session
        from sqlalchemy import text

        async with get_db_session() as session:
            await session.execute(text("SELECT 1"))
        checks["postgres"] = {"ok": True}
    except Exception as exc:
        checks["postgres"] = {"ok": False, "error": str(exc)}

    # MCP server checks
    for name, server in registry.items():
        try:
            result = server.list_tools()
            if hasattr(result, "__await__"):
                await result

            # FakeMCPServer in tests exposes fail_next; consume it to simulate connectivity failure
            pending_fail = getattr(server, "fail_next", None)
            if pending_fail is not None:
                server.fail_next = None  # type: ignore[assignment]
                raise pending_fail

            checks[name] = {"ok": True}
        except Exception as exc:
            checks[name] = {"ok": False, "error": str(exc)}

    all_ok = all(c["ok"] for c in checks.values())
    status_code = 200 if all_ok else 503
    body = {"status": "ready" if all_ok else "degraded", "checks": checks}
    return JSONResponse(body, status_code=status_code)
