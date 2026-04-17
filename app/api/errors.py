"""RFC 7807 problem+json error helpers."""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse

PROBLEM_CONTENT_TYPE = "application/problem+json"


def problem(
    status: int,
    title: str,
    detail: str = "",
    type_uri: str = "about:blank",
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": type_uri,
        "title": title,
        "status": status,
        "detail": detail,
    }
    if extra:
        body.update(extra)
    return JSONResponse(
        content=body,
        status_code=status,
        media_type=PROBLEM_CONTENT_TYPE,
    )


def not_found(detail: str) -> JSONResponse:
    return problem(404, "Not Found", detail)


def bad_request(detail: str) -> JSONResponse:
    return problem(400, "Bad Request", detail)


def unprocessable(detail: str, errors: list[dict[str, Any]] | None = None) -> JSONResponse:
    extra = {"errors": errors} if errors else {}
    return problem(422, "Unprocessable Entity", detail, extra=extra)


def internal_error() -> JSONResponse:
    return problem(500, "Internal Server Error", "An unexpected error occurred.")
