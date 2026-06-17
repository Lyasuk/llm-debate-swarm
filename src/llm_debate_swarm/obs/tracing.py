"""OpenTelemetry tracing for the debate-swarm engine.

Instrumentation is a **no-op until `setup_tracing()` is called**, so it never
affects normal runs or tests. Once enabled:

- spans are captured in-process and can be dumped to JSON (used to render the
  trace diagram in the README), and
- if ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, spans are ALSO exported over OTLP to
  any backend — Langfuse, Jaeger, Grafana Tempo, etc. (install the optional
  ``otlp`` extra). The engine is thus vendor-neutral: one env var points it at a
  real observability dashboard.
"""
from __future__ import annotations

import json
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


class _CaptureExporter(SpanExporter):
    """Keeps finished spans in memory so they can be dumped/rendered."""

    def __init__(self) -> None:
        self.spans: list[dict] = []

    def export(self, spans) -> SpanExportResult:
        for s in spans:
            ctx = s.get_span_context()
            parent_id = s.parent.span_id if s.parent else None
            self.spans.append({
                "name": s.name,
                "span_id": format(ctx.span_id, "016x"),
                "trace_id": format(ctx.trace_id, "032x"),
                "parent_id": format(parent_id, "016x") if parent_id else None,
                "start_ns": s.start_time,
                "end_ns": s.end_time,
                "duration_ms": round((s.end_time - s.start_time) / 1e6, 1),
                "status": s.status.status_code.name,
                "attributes": {k: v for k, v in (s.attributes or {}).items()},
            })
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover
        pass


_capture: _CaptureExporter | None = None


def setup_tracing(service_name: str = "llm-debate-swarm") -> trace.Tracer:
    """Install a TracerProvider that captures spans (and optionally exports OTLP)."""
    global _capture
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    _capture = _CaptureExporter()
    provider.add_span_processor(SimpleSpanProcessor(_capture))

    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
        except ImportError:
            import warnings

            warnings.warn(
                "OTEL_EXPORTER_OTLP_ENDPOINT is set but the OTLP exporter is not "
                "installed. Run: pip install 'llm-debate-swarm[otlp]'."
            )

    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


def get_tracer(name: str = "llm-debate-swarm") -> trace.Tracer:
    """Return a tracer. A no-op until ``setup_tracing()`` has been called."""
    return trace.get_tracer(name)


def dump_spans(path: str) -> int:
    """Write captured spans to JSON; returns how many were written."""
    spans = _capture.spans if _capture else []
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spans, f, indent=2, default=str)
    return len(spans)
