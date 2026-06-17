"""llm-debate-swarm — multi-LLM debate & arbitration engine.

A weighted multi-provider LLM consensus plus an N-agent, multi-round debiasing
debate swarm, aggregated with robust statistics into a single calibrated
probability with a confidence and disagreement signal.
"""
from llm_debate_swarm.engine import DebateSwarmEngine
from llm_debate_swarm.types import Question, Verdict

__version__ = "0.1.0"
__all__ = ["DebateSwarmEngine", "Question", "Verdict", "__version__"]
