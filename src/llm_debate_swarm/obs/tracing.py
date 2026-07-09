"""OpenTelemetry tracing for the debate-swarm engine.

Instrumentation is a **no-op until `setup_tracing()` is called**, so it never
affects normal runs or tests. Once enabled:

- spans are captured in-process and can be dumped to JSON (used to render the
  trace diagram in the README), and
- the SAME spans are fanned out, in-process, to any configured OTLP backend.
  We write the instrumentation **once** and route it to multiple dashboards —
  that is the vendor-neutral point:

  * **Langfuse** (self-host, privacy/system-of-record) — set ``LANGFUSE_HOST``,
    ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``.
  * **LangSmith** (managed, native LangGraph trace UI) — set
    ``LANGSMITH_API_KEY`` (+ optional ``LANGSMITH_PROJECT``).
  * any other OTLP collector — set ``OTEL_EXPORTER_OTLP_ENDPOINT``.

  All require the optional ``otlp`` extra (``pip install 'llm-debate-swarm[otlp]'``).
  At scale the recommended shape is one OpenTelemetry Collector fanning out to
  both backends (redaction/sampling in one place) — see ``deploy/otel-collector``;
  the in-process dual-export here needs zero extra infrastructure.

Per-LLM-call spans carry the OTel GenAI semantic-convention attributes
(``gen_ai.request.model``, ``gen_ai.usage.input_tokens`` / ``output_tokens``,
``gen_ai.provider.name``) so Langfuse/LangSmith infer model + token cost
natively, plus an explicit ``llm.cost_usd`` and domain attributes (role,
question id, persona, debate round).
"""
from __future__ import annotations

import base64
import json
import os
import warnings

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
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


def _otlp_exporter(endpoint: str, headers: dict[str, str]):
    """Build an OTLP/HTTP span exporter, or return None if the extra is missing."""
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        warnings.warn(
            "An OTLP backend is configured but the exporter is not installed. "
            "Run: pip install 'llm-debate-swarm[otlp]'."
        )
        return None
    return OTLPSpanExporter(endpoint=endpoint, headers=headers)


def _add_backends(provider: TracerProvider) -> list[str]:
    """Register OTLP exporters for every backend configured via env vars.

    One instrumentation, many dashboards. Returns the list of enabled backends
    (handy for logging/tests).
    """
    enabled: list[str] = []

    # Langfuse (self-hosted or cloud) — OTLP/HTTP + Basic auth (base64 pk:sk).
    lf_host = os.environ.get("LANGFUSE_HOST")
    lf_pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
    lf_sk = os.environ.get("LANGFUSE_SECRET_KEY")
    if lf_host and lf_pk and lf_sk:
        auth = base64.b64encode(f"{lf_pk}:{lf_sk}".encode()).decode()
        exp = _otlp_exporter(
            lf_host.rstrip("/") + "/api/public/otel/v1/traces",
            {"Authorization": f"Basic {auth}"},
        )
        if exp is not None:
            provider.add_span_processor(BatchSpanProcessor(exp))
            enabled.append("langfuse")

    # LangSmith — OTLP receiver + x-api-key header.
    ls_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if ls_key:
        endpoint = os.environ.get(
            "LANGSMITH_OTEL_ENDPOINT", "https://api.smith.langchain.com/otel/v1/traces"
        )
        headers = {"x-api-key": ls_key}
        project = os.environ.get("LANGSMITH_PROJECT")
        if project:
            headers["Langsmith-Project"] = project
        exp = _otlp_exporter(endpoint, headers)
        if exp is not None:
            provider.add_span_processor(BatchSpanProcessor(exp))
            enabled.append("langsmith")

    # Generic OTLP collector (Jaeger/Tempo/your own fan-out collector).
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        exp = _otlp_exporter(
            os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"].rstrip("/") + "/v1/traces", {}
        )
        if exp is not None:
            provider.add_span_processor(BatchSpanProcessor(exp))
            enabled.append("otlp")

    return enabled


def setup_tracing(service_name: str = "llm-debate-swarm") -> trace.Tracer:
    """Install a TracerProvider that captures spans (and optionally exports OTLP).

    Idempotent: repeated calls in the same process reuse the first capturing
    provider (OpenTelemetry forbids overriding an already-set TracerProvider).
    """
    global _capture
    existing = trace.get_tracer_provider()
    if isinstance(existing, TracerProvider) and _capture is not None:
        return trace.get_tracer(service_name)
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    _capture = _CaptureExporter()
    provider.add_span_processor(SimpleSpanProcessor(_capture))
    _add_backends(provider)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


def get_tracer(name: str = "llm-debate-swarm") -> trace.Tracer:
    """Return a tracer. A no-op until ``setup_tracing()`` has been called."""
    return trace.get_tracer(name)


def record_llm_span(
    span,
    *,
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: float | None = None,
    role: str = "",
    question_id: str | None = None,
    persona: str | None = None,
    debate_round: int | None = None,
    cost_usd: float | None = None,
) -> float:
    """Attach GenAI + cost + domain attributes to an LLM-call span.

    Uses the OTel GenAI semantic-convention attribute names so Langfuse and
    LangSmith parse model/usage/cost natively; also sets ``llm.*`` aliases the
    in-repo SVG renderer reads. Cost is computed from the shared
    :data:`~llm_debate_swarm.utils.cost_tracker.MODEL_PRICING` table unless a
    provider-reported ``cost_usd`` is passed. Returns the cost in USD.
    """
    if cost_usd is None:
        from llm_debate_swarm.utils.cost_tracker import get_tracker

        cost_usd = get_tracker().compute_cost(model, input_tokens, output_tokens)

    # OTel GenAI semantic conventions (parsed natively by Langfuse/LangSmith).
    span.set_attribute("gen_ai.provider.name", provider)
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
    span.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))
    # Aliases the SVG renderer + our metrics read.
    span.set_attribute("llm.model", model)
    span.set_attribute("llm.provider", provider)
    span.set_attribute("llm.cost_usd", float(cost_usd))
    if latency_ms is not None:
        span.set_attribute("llm.latency_ms", round(float(latency_ms), 1))
    if role:
        span.set_attribute("llm.role", role)
    if question_id is not None:
        span.set_attribute("forecast.question_id", str(question_id))
    if persona is not None:
        span.set_attribute("swarm.persona", str(persona))
    if debate_round is not None:
        span.set_attribute("swarm.round", int(debate_round))
    return float(cost_usd)


def dump_spans(path: str) -> int:
    """Write captured spans to JSON; returns how many were written."""
    spans = _capture.spans if _capture else []
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spans, f, indent=2, default=str)
    return len(spans)


def captured_spans() -> list[dict]:
    """Return the spans captured so far (empty until ``setup_tracing``)."""
    return _capture.spans if _capture else []
