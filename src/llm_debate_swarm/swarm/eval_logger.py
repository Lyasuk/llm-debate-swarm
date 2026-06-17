"""JSONL evaluation logger for swarm A/B testing."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from llm_debate_swarm.types import Question as Market
from llm_debate_swarm.swarm.simulator import SwarmResult
from llm_debate_swarm.utils.logger import get_logger

log = get_logger("swarm.eval")


class SwarmEvalLogger:
    """Logs detailed swarm simulation data to JSONL for post-hoc analysis."""

    def __init__(self, data_dir: str = "data/swarm_eval"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "simulations.jsonl"

    def log(
        self,
        market: Market,
        llm_prob: float,
        swarm_result: SwarmResult,
        blended_prob: float | None = None,
    ) -> None:
        """Append one simulation record to JSONL."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "market_id": market.id,
            "market_question": market.question[:200],
            "market_category": market.category,
            "market_price": market.yes_price,
            # Model tracking for A/B: GPT-4o-mini vs Gemini Flash-Lite
            "swarm_model": swarm_result.swarm_model if hasattr(swarm_result, "swarm_model") else "unknown",
            # Probabilities
            "llm_probability": round(llm_prob, 4),
            "swarm_probability": round(swarm_result.probability, 4),
            "blended_probability": round(blended_prob, 4) if blended_prob else None,
            "raw_trimmed_mean": round(swarm_result.raw_trimmed_mean, 4),
            # Debiasing metrics
            "blind_mean": round(swarm_result.blind_mean, 4),
            "aware_mean": round(swarm_result.aware_mean, 4),
            "anchoring_shift": round(swarm_result.anchoring_shift, 4),
            "convergence_ratio": round(swarm_result.convergence_ratio, 4),
            "premortem_changed_frac": round(swarm_result.premortem_changed_frac, 4),
            # Per-round std (convergence analysis)
            "std_per_round": [round(s, 4) for s in swarm_result.std_per_round],
            # Cost and performance
            "cost_usd": round(swarm_result.cost_usd, 5),
            "duration_sec": round(swarm_result.duration_sec, 1),
            "agent_count": swarm_result.agent_count,
            "rounds_completed": swarm_result.rounds_completed,
            # Meta-synthesis bias corrections
            "meta_bias_corrections": swarm_result.meta_bias_corrections,
            # All agent finals (for detailed analysis)
            "agent_finals": swarm_result.agent_finals,
            # Agreement
            "agreement": round(1.0 - abs(swarm_result.probability - llm_prob), 4),
            # Error
            "error": swarm_result.error,
        }

        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")
            log.info(f"Logged swarm eval: {market.question[:50]}...")
        except Exception as exc:
            log.warning(f"Failed to log swarm eval: {exc}")
