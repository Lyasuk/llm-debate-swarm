"""Import-graph smoke test: proves the engine de-couples cleanly from the
exchange/trading layer it was extracted from (no dangling imports)."""
import importlib

import pytest

MODULES = [
    "llm_debate_swarm",
    "llm_debate_swarm.engine",
    "llm_debate_swarm.config",
    "llm_debate_swarm.types",
    "llm_debate_swarm.cli",
    "llm_debate_swarm.swarm.simulator",
    "llm_debate_swarm.swarm.aggregator",
    "llm_debate_swarm.swarm.statistical_meta",
    "llm_debate_swarm.analysis.multi_llm_analyzer",
    "llm_debate_swarm.analysis.question_classifier",
    "llm_debate_swarm.research.research_manager",
    "llm_debate_swarm.tracking.decision_logger",
    "llm_debate_swarm.utils.cost_tracker",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def test_engine_constructs_without_network():
    # Build the engine with both LLM stages disabled — no provider calls, no keys.
    from llm_debate_swarm import DebateSwarmEngine

    engine = DebateSwarmEngine(use_consensus=False, use_swarm=False)
    assert engine.analyzer is None
    assert engine.swarm is None
