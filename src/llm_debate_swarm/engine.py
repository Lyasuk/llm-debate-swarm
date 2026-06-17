"""DebateSwarmEngine — domain-neutral multi-LLM debate & arbitration engine.

Pipeline::

    question
      -> classify (rule-based question typing)
      -> (optional) web research
      -> [ multi-LLM weighted consensus  ||  N-agent debiasing debate swarm ]  (parallel)
      -> combine into a calibrated Verdict

Both stages are optional and degrade gracefully: if one fails or is disabled,
the other still produces a verdict.
"""
from __future__ import annotations

import asyncio

from llm_debate_swarm.analysis.multi_llm_analyzer import MultiLLMAnalyzer
from llm_debate_swarm.analysis.question_classifier import classify_question
from llm_debate_swarm.config import AppConfig, load_config
from llm_debate_swarm.swarm import SwarmSimulator
from llm_debate_swarm.types import Question, Verdict
from llm_debate_swarm.utils.logger import get_logger

log = get_logger("engine")


class DebateSwarmEngine:
    """Orchestrates a multi-LLM consensus and a debate swarm into one verdict.

    Args:
        config: an :class:`AppConfig`; loaded from ``config.yaml`` when omitted.
        use_consensus: run the weighted multi-provider consensus stage.
        use_swarm: run the N-agent debiasing debate swarm stage.
        research: enable Tavily web research before forecasting (needs a key).
        swarm_weight: weight of the swarm probability when combining the two
            stages (0 = consensus only, 1 = swarm only, default 0.5).
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        use_consensus: bool = True,
        use_swarm: bool = True,
        research: bool = False,
        swarm_weight: float = 0.5,
    ) -> None:
        self.config = config or load_config()
        self.swarm_weight = swarm_weight
        self.analyzer = MultiLLMAnalyzer(self.config) if use_consensus else None
        self.swarm = SwarmSimulator(self.config) if use_swarm else None
        self.research_mgr = None
        self.doc_compiler = None
        if research:
            from llm_debate_swarm.research.document_compiler import DocumentCompiler
            from llm_debate_swarm.research.research_manager import ResearchManager

            self.research_mgr = ResearchManager(self.config)
            self.doc_compiler = DocumentCompiler(self.config)

    async def forecast(
        self,
        question: str,
        *,
        category: str = "",
        prior: float | None = None,
        horizon_days: float | None = None,
    ) -> Verdict:
        """Forecast a single binary question and return a calibrated verdict."""
        q = Question(
            question=question,
            category=category,
            prior=prior,
            horizon_days=horizon_days,
        )
        classification = classify_question(question, category)

        research_doc = ""
        if self.research_mgr is not None and self.doc_compiler is not None:
            try:
                ctx = await self.research_mgr.research_market(q, classification=classification)
                research_doc = self.doc_compiler.compile(ctx)
            except Exception as exc:  # research is best-effort
                log.warning(f"Research failed, continuing without it: {exc}")

        labels: list[str] = []
        tasks: list = []
        if self.analyzer is not None:
            labels.append("consensus")
            tasks.append(self.analyzer.analyze(q, research_doc, ""))
        if self.swarm is not None:
            labels.append("swarm")
            tasks.append(self.swarm.simulate(q, research_doc, classification))

        if not tasks:
            raise ValueError("Both stages are disabled — nothing to run.")

        results = await asyncio.gather(*tasks, return_exceptions=True)
        by = dict(zip(labels, results))

        consensus = by.get("consensus")
        if isinstance(consensus, Exception):
            log.warning(f"Consensus stage failed: {consensus}")
            consensus = None

        swarm = by.get("swarm")
        if isinstance(swarm, Exception):
            log.warning(f"Swarm stage failed: {swarm}")
            swarm = None
        if swarm is not None and getattr(swarm, "error", None):
            log.warning(f"Swarm error: {swarm.error}")
            swarm = None

        c_prob = consensus.consensus_probability if (consensus and consensus.is_valid) else None
        s_prob = swarm.probability if swarm is not None else None

        if c_prob is None and s_prob is None:
            raise RuntimeError("Both consensus and swarm failed — no forecast produced.")

        if c_prob is not None and s_prob is not None:
            w = self.swarm_weight
            probability = (1.0 - w) * c_prob + w * s_prob
            disagreement = abs(c_prob - s_prob)
            base_conf = (1.0 - w) * consensus.confidence + w * swarm.confidence
            confidence = max(0.0, base_conf * (1.0 - disagreement))
        elif s_prob is not None:
            probability, confidence, disagreement = s_prob, swarm.confidence, 0.0
        else:
            probability, confidence, disagreement = c_prob, consensus.confidence, 0.0

        per_model: list[dict] = []
        if consensus is not None:
            for p in consensus.predictions:
                per_model.append({
                    "model": p.model_name,
                    "provider": p.provider,
                    "probability": p.probability,
                    "confidence": p.confidence,
                    "error": p.error or "",
                })

        return Verdict(
            question=question,
            probability=probability,
            confidence=confidence,
            consensus_probability=c_prob,
            swarm_probability=s_prob,
            disagreement=disagreement,
            consensus_spread=consensus.spread if consensus else 0.0,
            anchoring_shift=swarm.anchoring_shift if swarm else 0.0,
            convergence_ratio=swarm.convergence_ratio if swarm else 1.0,
            agent_count=swarm.agent_count if swarm else 0,
            rounds_completed=swarm.rounds_completed if swarm else 0,
            models_responded=consensus.models_responded if consensus else 0,
            cost_usd=(swarm.cost_usd if swarm else 0.0),
            question_type=classification.question_type.value,
            per_model=per_model,
        )
