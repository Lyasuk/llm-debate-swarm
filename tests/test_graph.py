import types

import pytest

from llm_debate_swarm.engine import combine_verdict


def _classification():
    return types.SimpleNamespace(
        question_type=types.SimpleNamespace(value="fixed_date_event")
    )


def _consensus(prob=0.6):
    return types.SimpleNamespace(
        is_valid=True, consensus_probability=prob, confidence=0.7, spread=0.2,
        models_responded=3, predictions=[],
    )


def _swarm(prob=0.4):
    return types.SimpleNamespace(
        probability=prob, confidence=0.6, anchoring_shift=0.1, convergence_ratio=0.9,
        agent_count=30, rounds_completed=7, cost_usd=0.05, error=None,
    )


def test_combine_consensus_only():
    v = combine_verdict("Q?", _consensus(0.6), None, _classification(), 0.5)
    assert abs(v.probability - 0.6) < 1e-9
    assert v.swarm_probability is None


def test_combine_blends_both_5050():
    v = combine_verdict("Q?", _consensus(0.6), _swarm(0.4), _classification(), 0.5)
    assert abs(v.probability - 0.5) < 1e-9          # 0.5*0.6 + 0.5*0.4
    assert abs(v.disagreement - 0.2) < 1e-9


def test_combine_requires_at_least_one():
    with pytest.raises(RuntimeError):
        combine_verdict("Q?", None, None, _classification())


def test_graph_compiles():
    pytest.importorskip("langgraph")
    from llm_debate_swarm.graph import build_forecast_graph

    graph = build_forecast_graph(use_consensus=True, use_swarm=False)
    assert graph is not None  # compiled StateGraph (classify -> consensus -> combine)
