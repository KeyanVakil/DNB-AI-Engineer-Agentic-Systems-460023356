"""Structured JSON logging with OTel trace correlation."""

from __future__ import annotations

import logging
import sys
from typing import Any

from opentelemetry import trace

_CONFIGURED = False


class _OTelFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = trace.get_current_span().get_span_context()
        record.trace_id = format(ctx.trace_id, "032x") if ctx and ctx.is_valid else ""  # type: ignore[attr-defined]
        record.span_id = format(ctx.span_id, "016x") if ctx and ctx.is_valid else ""  # type: ignore[attr-defined]
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        import json

        payload: dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", ""),
            "span_id": getattr(record, "span_id", ""),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        skip = set(logging.LogRecord.__dict__) | set(payload) | {"trace_id", "span_id", "args"}
        extra = {
            k: v
            for k, v in record.__dict__.items()
            if k not in skip and not k.startswith("_")
        }
        payload.update(extra)
        return json.dumps(payload)


def configure_logging(level: str = "info") -> None:
    global _CONFIGURED
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(_OTelFilter())
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger that always emits JSON with OTel trace/span IDs."""
    logger = logging.getLogger(name)

    # Ensure this specific logger has a direct handler so it's not blocked by root level
    if not any(getattr(h, "_finsight_direct", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler._finsight_direct = True  # type: ignore[attr-defined]
        handler.addFilter(_OTelFilter())
        handler.setFormatter(_JsonFormatter())
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    return logger
