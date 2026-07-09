"""End-to-end tests for the primary LangGraph orchestration — no network.

Covers the idioms that make the graph real rather than decorative: Send fan-out,
the reducer that merges parallel debaters, durable execution (kill/resume from a
checkpoint), and the per-run timeout budget.
"""
from __future__ import annotations

import asyncio

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from llm_debate_swarm.analysis.multi_llm_analyzer import LLMPrediction, MultiLLMAnalyzer
from llm_debate_swarm.graph import build_forecast_graph, forecast_with_graph


def _fake_pred(model_config, prob=0.6):
    return LLMPrediction(
        model_name=model_config.name, provider=model_config.provider,
        probability=prob, confidence="high", input_tokens=10, output_tokens=5,
    )


def test_graph_runs_and_reducer_merges_all_debaters(monkeypatch):
    async def fake_query_model(self, model_config, prompt, *, question_id=None):
        return _fake_pred(model_config)

    monkeypatch.setattr(MultiLLMAnalyzer, "_query_model", fake_query_model)
    graph = build_forecast_graph(use_swarm=False)

    out = asyncio.run(graph.ainvoke({"question": "Will X happen?"}))
    v = out["verdict"]
    assert v is not None
    assert abs(v.probability - 0.6) < 1e-6
    # Reducer proof: every configured model's prediction survived the parallel
    # Send fan-out. Last-write-wins (a missing reducer) would give 1, not N.
    assert v.models_responded >= 2
    assert v.models_responded == len(v.per_model)


def test_per_run_timeout_budget_is_enforced(monkeypatch):
    async def slow_query_model(self, model_config, prompt, *, question_id=None):
        await asyncio.sleep(0.5)
        return _fake_pred(model_config)

    monkeypatch.setattr(MultiLLMAnalyzer, "_query_model", slow_query_model)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(forecast_with_graph("Q?", use_swarm=False, timeout_sec=0.05))


def test_durable_execution_resumes_without_rerunning_debaters(monkeypatch):
    calls = {"debater": 0, "combine": 0}

    async def counting_query_model(self, model_config, prompt, *, question_id=None):
        calls["debater"] += 1
        return _fake_pred(model_config)

    from llm_debate_swarm.graph import combine_verdict as real_combine

    def flaky_combine(*a, **k):
        calls["combine"] += 1
        if calls["combine"] == 1:
            raise RuntimeError("boom in combine")
        return real_combine(*a, **k)

    monkeypatch.setattr(MultiLLMAnalyzer, "_query_model", counting_query_model)
    monkeypatch.setattr("llm_debate_swarm.graph.combine_verdict", flaky_combine)

    graph = build_forecast_graph(use_swarm=False, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-recover"}, "recursion_limit": 25}

    # First run dies in `combine` (debaters + aggregate already checkpointed).
    with pytest.raises(Exception):
        asyncio.run(graph.ainvoke({"question": "Q?"}, config=cfg))
    debaters_first = calls["debater"]
    assert debaters_first >= 2

    # Resume same thread_id: only `combine` re-runs; debaters are NOT re-queried.
    out = asyncio.run(graph.ainvoke(None, config=cfg))
    assert out["verdict"] is not None
    assert calls["debater"] == debaters_first  # durable execution proven
    assert calls["combine"] == 2
