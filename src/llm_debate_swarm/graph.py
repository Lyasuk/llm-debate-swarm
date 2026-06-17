"""LangGraph orchestration of the debate-swarm engine.

Expresses the same pipeline as :meth:`DebateSwarmEngine.forecast` — classify, then
a **fan-out** to the multi-LLM consensus and the debate swarm running in parallel,
then a **fan-in** that fuses them — as an explicit LangGraph ``StateGraph``::

            ┌─> consensus ─┐
    classify              ├─> combine
            └─> swarm ─────┘

It is an alternative orchestration surface over the SAME engine components (it
reuses the analyzer, swarm, classifier, and `combine_verdict`), for teams whose
stack is standardized on LangGraph.

Requires the optional extra::

    pip install 'llm-debate-swarm[graph]'
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from llm_debate_swarm.analysis.multi_llm_analyzer import MultiLLMAnalyzer
from llm_debate_swarm.analysis.question_classifier import classify_question
from llm_debate_swarm.config import AppConfig, load_config
from llm_debate_swarm.engine import combine_verdict
from llm_debate_swarm.swarm import SwarmSimulator
from llm_debate_swarm.types import Question, Verdict


class ForecastState(TypedDict, total=False):
    """Graph state. ``consensus`` and ``swarm`` are written by separate parallel
    nodes (distinct keys, so no reducer is needed)."""

    question: str
    category: str
    classification: Any
    consensus: Any
    swarm: Any
    verdict: Verdict


def build_forecast_graph(
    config: AppConfig | None = None,
    *,
    use_consensus: bool = True,
    use_swarm: bool = True,
    swarm_weight: float = 0.5,
):
    """Build and compile the LangGraph forecast graph over the engine components."""
    config = config or load_config()
    analyzer = MultiLLMAnalyzer(config) if use_consensus else None
    swarm = SwarmSimulator(config) if use_swarm else None

    async def classify_node(state: ForecastState) -> dict:
        c = classify_question(state["question"], state.get("category", ""))
        return {"classification": c}

    async def consensus_node(state: ForecastState) -> dict:
        q = Question(question=state["question"], category=state.get("category", ""))
        return {"consensus": await analyzer.analyze(q, "", "")}

    async def swarm_node(state: ForecastState) -> dict:
        q = Question(question=state["question"], category=state.get("category", ""))
        return {"swarm": await swarm.simulate(q, "", state["classification"])}

    async def combine_node(state: ForecastState) -> dict:
        verdict = combine_verdict(
            state["question"], state.get("consensus"), state.get("swarm"),
            state["classification"], swarm_weight,
        )
        return {"verdict": verdict}

    g = StateGraph(ForecastState)
    g.add_node("classify", classify_node)
    g.add_node("combine", combine_node)
    g.add_edge(START, "classify")

    if analyzer is not None:
        g.add_node("consensus", consensus_node)
        g.add_edge("classify", "consensus")   # fan-out
        g.add_edge("consensus", "combine")     # fan-in
    if swarm is not None:
        g.add_node("swarm", swarm_node)
        g.add_edge("classify", "swarm")        # fan-out
        g.add_edge("swarm", "combine")         # fan-in

    g.add_edge("combine", END)
    return g.compile()


async def forecast_with_graph(question: str, **kwargs) -> Verdict:
    """Convenience: build the graph and run one forecast through it."""
    graph = build_forecast_graph(**kwargs)
    out = await graph.ainvoke({"question": question})
    return out["verdict"]
