"""OpenTelemetry SDK setup and in-memory exporter helpers."""

from __future__ import annotations

import threading
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

_provider: TracerProvider | None = None
_memory_processors: dict[int, "_BufferExporter"] = {}
_lock = threading.Lock()

# Span storage keyed by trace_id for API retrieval
_trace_store: dict[str, list[dict[str, Any]]] = {}
_run_trace_map: dict[str, str] = {}  # run_id -> trace_id


class _TraceStoreExporter(SpanExporter):
    """Always-on exporter: fills _trace_store so spans survive beyond per-test buffers."""

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:  # type: ignore[override]
        for span in spans:
            span_dict = _span_to_dict(span)
            tid = span_dict.get("trace_id", "")
            if tid:
                _trace_store.setdefault(tid, []).append(span_dict)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def setup_otel(service_name: str = "finsight-agents") -> TracerProvider:
    global _provider
    if _provider is not None:
        return _provider

    _provider = TracerProvider()
    trace.set_tracer_provider(_provider)
    # Always-on collector so _trace_store is filled even before per-test exporters
    _provider.add_span_processor(SimpleSpanProcessor(_TraceStoreExporter()))
    return _provider


def get_tracer(name: str = "finsight") -> trace.Tracer:
    if _provider is None:
        setup_otel()
    return trace.get_tracer(name)


class _BufferExporter(SpanExporter):
    def __init__(self, buffer: list[dict[str, Any]]) -> None:
        self._buffer = buffer

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:  # type: ignore[override]
        for span in spans:
            self._buffer.append(_span_to_dict(span))
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def install_memory_exporter(buffer: list[dict[str, Any]]) -> int:
    global _provider
    if _provider is None:
        setup_otel()
    assert _provider is not None
    # Pre-fill with all existing spans so tests using completed_run fixture see them
    for span_list in _trace_store.values():
        buffer.extend(span_list)
    exporter = _BufferExporter(buffer)
    processor = SimpleSpanProcessor(exporter)
    _provider.add_span_processor(processor)
    token = id(processor)
    with _lock:
        _memory_processors[token] = exporter
    return token


def uninstall_memory_exporter(token: int) -> None:
    with _lock:
        _memory_processors.pop(token, None)


def _span_to_dict(span: ReadableSpan) -> dict[str, Any]:
    ctx = span.context
    parent_ctx = span.parent

    trace_id = format(ctx.trace_id, "032x") if ctx else ""
    span_id = format(ctx.span_id, "016x") if ctx else ""
    parent_span_id = format(parent_ctx.span_id, "016x") if parent_ctx else None

    attrs = dict(span.attributes or {})

    kind_map = {
        0: "internal",
        1: "server",
        2: "client",
        3: "producer",
        4: "consumer",
    }
    kind = kind_map.get(span.kind.value, "internal")

    return {
        "name": span.name,
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "kind": kind,
        "attributes": attrs,
        "children": [],
    }


def build_trace_tree(trace_id: str) -> dict[str, Any] | None:
    spans = _trace_store.get(trace_id)
    if not spans:
        return None
    return _build_tree(spans)


def _build_tree(spans: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str | None, dict[str, Any]] = {s["span_id"]: dict(s) for s in spans}
    for s in by_id.values():
        s["children"] = []
    roots: list[dict[str, Any]] = []
    for span in by_id.values():
        parent_id = span.get("parent_span_id")
        if parent_id and parent_id in by_id:
            by_id[parent_id]["children"].append(span)
        else:
            roots.append(span)
    return roots[0] if roots else {"name": "root", "trace_id": "", "children": list(by_id.values())}


def associate_run_with_trace(run_id: str, trace_id: str) -> None:
    _run_trace_map[run_id] = trace_id


def get_trace_for_run(run_id: str) -> dict[str, Any] | None:
    trace_id = _run_trace_map.get(run_id)
    if not trace_id:
        return None
    return build_trace_tree(trace_id)
