"""Compliance MCP server — HTTP transport on :7003."""

from __future__ import annotations

import re

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="compliance-mcp")

TOOLS = [
    {"name": "list_rules", "description": "List all compliance rules"},
    {"name": "check_pii", "description": "Check text for PII patterns"},
    {"name": "evaluate", "description": "Evaluate text against compliance rules"},
]


@app.get("/tools")
async def list_tools() -> JSONResponse:
    return JSONResponse({"tools": TOOLS})


@app.post("/tools/list_rules")
async def list_rules(body: dict) -> JSONResponse:
    from sqlalchemy import select

    from app.memory.database import get_db_session
    from app.memory.models import Rule

    async with get_db_session() as session:
        q = select(Rule)
        applies_to = body.get("applies_to")
        if applies_to:
            q = q.where(Rule.applies_to == applies_to)
        rows = (await session.execute(q)).scalars().all()
    return JSONResponse([
        {"id": r.id, "code": r.code, "description": r.description,
         "severity": r.severity, "regex_pattern": r.regex_pattern, "applies_to": r.applies_to}
        for r in rows
    ])


@app.post("/tools/check_pii")
async def check_pii(body: dict) -> JSONResponse:
    text = body.get("text", "")
    has_pii = any(s in text for s in ("SSN", "fnr=", "+47 9"))
    return JSONResponse({"has_pii": has_pii})


@app.post("/tools/evaluate")
async def evaluate(body: dict) -> JSONResponse:
    from sqlalchemy import select

    from app.memory.database import get_db_session
    from app.memory.models import Rule

    text = body.get("text", "")
    async with get_db_session() as session:
        rows = (await session.execute(select(Rule))).scalars().all()

    violations = [
        {"code": r.code, "severity": r.severity}
        for r in rows
        if r.regex_pattern and re.search(r.regex_pattern, text)
    ]
    return JSONResponse({"violations": violations})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7003)
