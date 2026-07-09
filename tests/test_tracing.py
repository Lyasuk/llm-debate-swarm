"""Tests for OpenTelemetry tracing: per-call GenAI attributes + dual export.

No network or API keys — exporters are configured but never flushed to a real
backend, and the LLM span is recorded with synthetic token counts.
"""
from __future__ import annotations

from opentelemetry.sdk.trace import TracerProvider

from llm_debate_swarm.obs.tracing import (
    _add_backends,
    captured_spans,
    record_llm_span,
    setup_tracing,
)


def test_dual_export_backends_registered(monkeypatch):
    """Langfuse + LangSmith env vars each register an OTLP exporter (no collector)."""
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    provider = TracerProvider()
    enabled = _add_backends(provider)

    assert "langfuse" in enabled
    assert "langsmith" in enabled


def test_llm_span_records_gen_ai_and_cost():
    """record_llm_span emits GenAI-convention attrs and a cost from the price table."""
    tracer = setup_tracing("test-trace")
    with tracer.start_as_current_span("llm.consensus") as span:
        cost = record_llm_span(
            span,
            provider="anthropic",
            model="claude-haiku-4-5-20251001",  # price (1.0, 5.0) per 1M tokens
            input_tokens=1000,
            output_tokens=200,
            latency_ms=1234.5,
            role="consensus",
            question_id="q-42",
        )

    # (1000 * 1.0 + 200 * 5.0) / 1e6 = 0.002
    assert cost == 0.002

    spans = [s for s in captured_spans() if s["name"] == "llm.consensus"]
    assert spans, "llm.consensus span was not captured"
    attrs = spans[-1]["attributes"]
    assert attrs["gen_ai.request.model"] == "claude-haiku-4-5-20251001"
    assert attrs["gen_ai.provider.name"] == "anthropic"
    assert attrs["gen_ai.usage.input_tokens"] == 1000
    assert attrs["gen_ai.usage.output_tokens"] == 200
    assert attrs["llm.cost_usd"] == 0.002
    assert attrs["llm.role"] == "consensus"
    assert attrs["forecast.question_id"] == "q-42"


def test_consensus_query_emits_priced_span(monkeypatch):
    """_query_model wraps a provider call in a priced llm.query span (no network)."""
    import asyncio

    from llm_debate_swarm.analysis.multi_llm_analyzer import (
        LLMPrediction,
        MultiLLMAnalyzer,
    )
    from llm_debate_swarm.config import LLMModelConfig, load_config

    setup_tracing("test-consensus")
    analyzer = MultiLLMAnalyzer(load_config())

    async def fake_anthropic(model, prompt):
        return LLMPrediction(
            model_name=model,
            provider="anthropic",
            probability=0.7,
            confidence="high",
            input_tokens=1000,
            output_tokens=200,
        )

    monkeypatch.setattr(analyzer, "_query_anthropic", fake_anthropic)
    mc = LLMModelConfig(
        name="claude-haiku-4-5-20251001", provider="anthropic", weight=1.0
    )
    pred = asyncio.run(analyzer._query_model(mc, "prompt", question_id="q-1"))

    assert pred.probability == 0.7
    spans = [s for s in captured_spans() if s["name"] == "llm.query"]
    assert spans, "llm.query span was not captured"
    attrs = spans[-1]["attributes"]
    assert attrs["gen_ai.usage.input_tokens"] == 1000
    assert attrs["gen_ai.usage.output_tokens"] == 200
    assert attrs["llm.cost_usd"] == 0.002
    assert attrs["llm.role"] == "consensus"
    assert attrs["llm.latency_ms"] >= 0
