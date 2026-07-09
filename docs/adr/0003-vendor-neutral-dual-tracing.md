# ADR-0003 — Vendor-neutral tracing → Langfuse + LangSmith

**Status:** Accepted · **Date:** 2026-07-09

## Context

Observability is the project's headline reliability signal: a reviewer must be
able to open one forecast and see the whole debate tree — every LLM call with its
model, tokens, cost, and latency. We don't want to marry one vendor's SDK.

## Options considered

- **A single vendor SDK** (LangSmith native, or Langfuse native callback) —
  richest for that vendor, but lock-in and non-portable spans.
- **OpenTelemetry, one instrumentation → many backends** — write spans once with
  the OTel GenAI semantic conventions (`gen_ai.request.model`,
  `gen_ai.usage.*`), export to any OTLP backend.
- **OTel Collector fan-out** — a collector duplicates spans to N exporters with
  redaction/sampling in one place.

## Decision

Instrument once with the OTel GenAI conventions and fan out **in-process** to both
**Langfuse** (self-host, open-source, privacy/system-of-record, owns the cost table
and the resolved-outcome scores) **and LangSmith** (managed, native LangGraph trace
UI, ATS/portfolio signal) via two OTLP exporters on the same TracerProvider. The
Collector fan-out (redaction/sampling in one place) is the documented at-scale
upgrade, not built for this size.

## Consequences

- **Pay:** redaction/sampling logic would be duplicated per exporter if we grew
  (which is exactly why the Collector is the scale-up path); GenAI semconv is still
  "experimental" so attribute names may shift.
- **Get:** zero vendor lock-in, both dashboards from one code path, Langfuse infers
  USD cost from `gen_ai.request.model` + `gen_ai.usage.*` natively — demonstrating
  the vendor-neutral point, which is the whole axis.

## Revisit when

- Trace volume forces central sampling/redaction → move to an OTel Collector.
- The OTel GenAI conventions stabilize (drop any compatibility shims).
