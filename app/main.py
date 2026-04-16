"""FastAPI application factory."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from opentelemetry import trace
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.errors import PROBLEM_CONTENT_TYPE, problem
from app.observability.otel import get_tracer, setup_otel


class _TraceparentMiddleware(BaseHTTPMiddleware):
    """Inject W3C traceparent header into every response and capture X-User."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        user = request.headers.get("X-User")
        tracer = get_tracer()
        span_name = f"{request.method} {request.url.path}"
        with tracer.start_as_current_span(
            span_name,
            kind=trace.SpanKind.SERVER,
        ) as span:
            span.set_attribute("http.route", str(request.url.path))
            span.set_attribute("http.method", request.method)
            if user:
                span.set_attribute("user.id", user)

            ctx = span.get_span_context()
            traceparent = ""
            if ctx and ctx.is_valid:
                trace_id = format(ctx.trace_id, "032x")
                span_id = format(ctx.span_id, "016x")
                flags = "01"
                traceparent = f"00-{trace_id}-{span_id}-{flags}"

            response = await call_next(request)
            if traceparent:
                response.headers["traceparent"] = traceparent
            return response


def create_app() -> FastAPI:
    setup_otel()

    from app.observability.logging import configure_logging
    from app.settings import get_settings

    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="FinSight Agents", version="0.1.0")
    app.add_middleware(_TraceparentMiddleware)

    # Mount routers
    from app.api import bench, customers, drift, health, reviews, runs

    app.include_router(health.router)
    app.include_router(runs.router)
    app.include_router(reviews.router)
    app.include_router(customers.router)
    app.include_router(bench.router)
    app.include_router(drift.router)

    # Prometheus /metrics endpoint
    @app.get("/metrics")
    async def metrics() -> Response:
        data = generate_latest()
        return Response(data, media_type=CONTENT_TYPE_LATEST)

    # Exception handlers — all errors are problem+json
    @app.exception_handler(HTTPException)
    async def http_exc_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return problem(exc.status_code, exc.detail or "HTTP Error", exc.detail or "")

    @app.exception_handler(RequestValidationError)
    async def validation_exc_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"field": str(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(
            {
                "type": "about:blank",
                "title": "Unprocessable Entity",
                "status": 422,
                "detail": "Validation error",
                "errors": errors,
            },
            status_code=422,
            media_type=PROBLEM_CONTENT_TYPE,
        )

    @app.exception_handler(405)
    async def method_not_allowed(request: Request, exc: Any) -> JSONResponse:
        return problem(405, "Method Not Allowed", f"{request.method} not allowed on {request.url.path}")

    @app.exception_handler(Exception)
    async def generic_exc_handler(request: Request, exc: Exception) -> JSONResponse:
        return problem(500, "Internal Server Error", "An unexpected error occurred.")

    return app
