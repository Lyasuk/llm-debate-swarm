"""Domain-neutral input/output types for the debate-swarm engine.

``Question`` is the generic unit the engine reasons about. It is intentionally
shaped to be a drop-in for the market object the swarm was originally built
around (same ``yes_price`` / ``no_price`` / ``days_to_resolution`` surface) but
carries no exchange-specific fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel


class Question(BaseModel):
    """A single binary (yes/no) question to forecast.

    Attributes:
        question: the natural-language question, e.g. "Will X happen by Y?".
        category: optional free-text category hint (improves classification).
        prior: optional market-/expert-implied P(YES) in [0, 1]. When unknown,
            the engine treats the question as a neutral 50/50 prior.
        horizon_days: optional number of days until the question resolves.
        resolution_source: optional note on how/where the question resolves.
    """

    question: str
    category: str = ""
    prior: float | None = None
    horizon_days: float | None = None
    resolution_source: str = ""

    model_config = {"populate_by_name": True}

    @property
    def yes_price(self) -> float:
        """Current implied P(YES); neutral 0.5 when no prior is supplied."""
        return self.prior if self.prior is not None else 0.5

    @property
    def no_price(self) -> float:
        return 1.0 - self.yes_price

    @property
    def days_to_resolution(self) -> float:
        """Days to resolution; a neutral 30.0 when no horizon is supplied."""
        return self.horizon_days if self.horizon_days is not None else 30.0


@dataclass
class Verdict:
    """Calibrated output of a debate-swarm run."""

    question: str
    probability: float            # combined P(YES) in [0, 1]
    confidence: float             # [0, 1]
    consensus_probability: float | None = None   # multi-LLM weighted consensus
    swarm_probability: float | None = None       # debate-swarm result
    disagreement: float = 0.0     # |consensus - swarm|
    consensus_spread: float = 0.0
    anchoring_shift: float = 0.0
    convergence_ratio: float = 1.0
    agent_count: int = 0
    rounds_completed: int = 0
    models_responded: int = 0
    cost_usd: float = 0.0
    question_type: str = "unknown"
    per_model: list[dict] = field(default_factory=list)
    notes: str = ""
