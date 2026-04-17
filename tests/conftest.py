"""Top-level pytest fixtures for the FinSight Agents test suite.

These fixtures wire together the pieces the integration tests need:

- A real Postgres instance (provided by docker compose or a spun-up testcontainer)
  into which the Alembic migrations and seed fixtures have been applied
- Fake MCP servers that replay canned responses so tests don't depend on real data
- A recording/replaying fake LLM that reads/writes JSON cassettes under
  `tests/fixtures/llm_cassettes/`
- An in-memory OTel span exporter so tests can assert on span structure
- A FastAPI test client wired to the real app factory but with the above fakes
  injected

All fixtures are async-aware; pytest-asyncio's `asyncio_mode = auto` is set in
`pytest.ini` so `async def test_*` functions don't need an explicit marker.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CASSETTES_DIR = FIXTURES_DIR / "llm_cassettes"


# ---------------------------------------------------------------------------
# Environment: every test runs in an isolated, deterministic config.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _test_env() -> Iterator[None]:
    """Set environment variables that the app reads on import.

    Using a session-scoped autouse fixture guarantees the app settings object
    sees the test values, not whatever is on the dev machine.
    """
    previous = dict(os.environ)
    os.environ.update(
        {
            "FINSIGHT_ENV": "test",
            "LLM_PROVIDER": "fake",
            "LLM_MODEL": "fake-llm",
            "DATABASE_URL": os.environ.get(
                "TEST_DATABASE_URL",
                "postgresql+asyncpg://finsight:finsight@localhost:5432/finsight_test",
            ),
            "MLFLOW_TRACKING_URI": "sqlite:///:memory:",
            "OTEL_SDK_DISABLED": "false",
            "OTEL_TRACES_EXPORTER": "memory",
            "OTEL_METRICS_EXPORTER": "memory",
            "CUSTOMER_MCP_URL": "fake://customer",
            "MARKET_MCP_URL": "fake://market",
            "COMPLIANCE_MCP_URL": "fake://compliance",
            "LOG_LEVEL": "warning",
        }
    )
    yield
    os.environ.clear()
    os.environ.update(previous)


# ---------------------------------------------------------------------------
# Database.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
async def db_engine():
    """Session-scoped async SQLAlchemy engine, migrated once and torn down at end."""
    from app.memory import database  # noqa: WPS433 - imported lazily on purpose

    engine = await database.create_engine_for_tests()
    await database.run_migrations(engine)
    try:
        yield engine
    finally:
        await database.drop_all(engine)
        await engine.dispose()


_APP_TABLES = (
    "memory_embeddings",
    "drift_events",
    "human_reviews",
    "eval_results",
    "runs",
    "holdings",
    "transactions",
    "accounts",
    "news",
    "prices",
    "rules",
    "customers",
)


async def _truncate_all(db_engine) -> None:
    from sqlalchemy import text

    async with db_engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE TABLE {', '.join(_APP_TABLES)} CASCADE"))


@pytest.fixture()
async def db_session(db_engine) -> AsyncIterator[Any]:
    """Function-scoped SQLAlchemy session. Commits are real (several
    integration tests drive the FastAPI app through HTTP, and its
    connection pool can't see rolled-back data) so we truncate all app
    tables before handing out the session.
    """
    from app.memory import database

    await _truncate_all(db_engine)
    factory = database.async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        yield session
        try:
            await session.commit()
        except Exception:
            await session.rollback()


@pytest.fixture()
async def seed_fixtures(db_session) -> dict[str, Any]:
    """Populate the three seeded domains (customers, market, compliance).

    Returns a dict with handy lookups (e.g. a known `customer_id`) so tests
    don't have to re-parse the JSON every time.
    """
    from app.memory import seeds

    payload = json.loads((FIXTURES_DIR / "customers.json").read_text(encoding="utf-8"))
    await seeds.load_customers(db_session, payload["customers"])
    await seeds.load_market(db_session, payload["market"])
    await seeds.load_compliance(db_session, payload["rules"])
    await db_session.commit()
    return payload


# ---------------------------------------------------------------------------
# MCP fakes.
# ---------------------------------------------------------------------------


class FakeMCPServer:
    """In-process MCP server that responds from a dict of tool handlers.

    This lets integration tests exercise the real MCP client wrapper without
    standing up subprocesses. It records every call for assertions.
    """

    def __init__(self, name: str, tools: dict[str, Any] | None = None) -> None:
        self.name = name
        self.tools: dict[str, Any] = tools or {}
        self.calls: list[dict[str, Any]] = []
        self.fail_next: Exception | None = None

    def register(self, tool_name: str, handler: Any) -> None:
        self.tools[tool_name] = handler

    async def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": name, "description": f"{self.name}.{name}"} for name in self.tools]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append({"tool": tool_name, "arguments": arguments})
        if self.fail_next is not None:
            err = self.fail_next
            self.fail_next = None
            raise err
        if tool_name not in self.tools:
            raise KeyError(f"unknown tool {tool_name!r} on {self.name!r}")
        handler = self.tools[tool_name]
        return handler(**arguments) if callable(handler) else handler


@pytest.fixture()
def fake_customer_mcp(seed_fixtures) -> FakeMCPServer:
    customers = {c["id"]: c for c in seed_fixtures["customers"]}
    server = FakeMCPServer("customer-mcp")
    server.register("get_profile", lambda customer_id: customers[customer_id]["profile"])
    server.register(
        "get_accounts", lambda customer_id: customers[customer_id]["accounts"]
    )
    server.register(
        "get_transactions",
        lambda customer_id, days=90: customers[customer_id]["transactions"][:200],
    )
    server.register(
        "get_holdings", lambda customer_id: customers[customer_id]["holdings"]
    )
    return server


@pytest.fixture()
def fake_market_mcp(seed_fixtures) -> FakeMCPServer:
    market = seed_fixtures["market"]
    server = FakeMCPServer("market-mcp")
    server.register("get_prices", lambda ticker, days=30: market["prices"].get(ticker, []))
    server.register("get_indices", lambda: market["indices"])
    server.register("get_news", lambda ticker, limit=5: market["news"].get(ticker, []))
    return server


@pytest.fixture()
def fake_compliance_mcp(seed_fixtures) -> FakeMCPServer:
    rules = seed_fixtures["rules"]
    server = FakeMCPServer("compliance-mcp")
    server.register("list_rules", lambda applies_to=None: rules)
    server.register(
        "check_pii",
        lambda text: {"has_pii": any(s in text for s in ("SSN", "fnr=", "+47 9"))},
    )
    server.register(
        "evaluate",
        lambda text, rule_codes=None: {
            "violations": [
                {"code": r["code"], "severity": r["severity"]}
                for r in rules
                if r.get("regex_pattern") and r["regex_pattern"] in text
            ]
        },
    )
    return server


@pytest.fixture()
def fake_mcp_registry(
    fake_customer_mcp, fake_market_mcp, fake_compliance_mcp
) -> dict[str, FakeMCPServer]:
    """The MCP client wrapper looks up servers by name from this registry."""
    return {
        "customer-mcp": fake_customer_mcp,
        "market-mcp": fake_market_mcp,
        "compliance-mcp": fake_compliance_mcp,
    }


# ---------------------------------------------------------------------------
# Fake LLM (record/replay cassettes).
# ---------------------------------------------------------------------------


class FakeLLM:
    """A deterministic stand-in for the real LLM provider.

    Loads responses from `tests/fixtures/llm_cassettes/<cassette>.json`.
    Each cassette is a list of `{prompt_hash, response}` records; the fake
    returns the first record whose prompt_hash matches. If no record matches
    and `strict=True`, the call raises, which keeps tests honest when prompts
    drift.
    """

    def __init__(self, cassette: str, strict: bool = True) -> None:
        self.cassette = cassette
        self.strict = strict
        self.calls: list[dict[str, Any]] = []
        path = CASSETTES_DIR / f"{cassette}.json"
        self._records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        call = {"model": model, "messages": messages, "kwargs": kwargs}
        self.calls.append(call)
        import hashlib

        prompt_hash = hashlib.sha256(
            json.dumps(messages, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        for record in self._records:
            if record.get("prompt_hash") in (prompt_hash, "*"):
                return {
                    "content": record["response"],
                    "usage": record.get("usage", {"prompt_tokens": 100, "completion_tokens": 80}),
                    "model": model,
                    "prompt_hash": prompt_hash,
                }
        if self.strict:
            raise AssertionError(
                f"No cassette match for prompt_hash={prompt_hash} in {self.cassette!r}"
            )
        return {
            "content": "",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "model": model,
            "prompt_hash": prompt_hash,
        }


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM(cassette="default", strict=False)


# ---------------------------------------------------------------------------
# OTel in-memory span exporter for assertions.
# ---------------------------------------------------------------------------


@pytest.fixture()
def otel_spans() -> Iterator[list[dict[str, Any]]]:
    """Install an in-memory span exporter and yield the captured spans list."""
    from app.observability import otel

    buffer: list[dict[str, Any]] = []
    token = otel.install_memory_exporter(buffer)
    try:
        yield buffer
    finally:
        otel.uninstall_memory_exporter(token)


# ---------------------------------------------------------------------------
# FastAPI test client with all fakes injected.
# ---------------------------------------------------------------------------


@pytest.fixture()
async def app(db_engine, fake_mcp_registry, fake_llm):
    from app.llm import provider as llm_provider
    from app.main import create_app
    from app.mcp import client as mcp_client

    application = create_app()
    application.dependency_overrides[mcp_client.get_registry] = lambda: fake_mcp_registry
    application.dependency_overrides[llm_provider.get_llm] = lambda: fake_llm
    return application


@pytest.fixture()
async def client(app) -> AsyncIterator[Any]:
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Convenience helpers.
# ---------------------------------------------------------------------------


@pytest.fixture()
def customer_id(seed_fixtures) -> str:
    return seed_fixtures["customers"][0]["id"]


@pytest.fixture()
def new_run_id() -> str:
    return str(uuid.uuid4())
