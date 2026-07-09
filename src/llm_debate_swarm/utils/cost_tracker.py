"""API cost tracking: per-call, per-model, per-cycle, per-decision.

Tracks:
- Anthropic (Claude Sonnet, Haiku, Opus)
- Google (Gemini Flash-Lite, Flash, Pro)
- OpenAI (GPT-4o, GPT-4o-mini) — legacy
- Tavily (web search)
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from llm_debate_swarm.utils.logger import get_logger

log = get_logger("cost_tracker")


# Cost per 1M tokens (input, output) in USD
# Updated 2026-04 pricing
MODEL_PRICING = {
    # Anthropic
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-3-5-haiku-20241022": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    # Google (per 1M)
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash-8b": (0.0375, 0.15),
    # OpenAI
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    # Tavily (flat fee per search, not per-token)
    "tavily-search": (0.008, 0.0),  # ~$8/1000 searches = $0.008 per search
    # Groq (free tier — tracking нульова вартість для accounting)
    "meta-llama/llama-4-scout-17b-16e-instruct": (0.0, 0.0),
    "llama-3.3-70b-versatile": (0.0, 0.0),
    "llama-3.1-8b-instant": (0.0, 0.0),
    "qwen/qwen3-32b": (0.0, 0.0),
    "openai/gpt-oss-20b": (0.0, 0.0),
    "openai/gpt-oss-120b": (0.0, 0.0),
}


@dataclass
class CostEntry:
    """Single API call cost entry."""
    timestamp: str
    provider: str
    model: str
    role: str  # consensus, swarm, meta, reeval, research
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cycle_id: Optional[str] = None
    decision_id: Optional[int] = None
    trade_id: Optional[int] = None


class CostTracker:
    """Thread-safe singleton cost tracker."""

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: list[CostEntry] = []
        self._current_cycle_id: Optional[str] = None
        self._current_decision_id: Optional[int] = None

    def set_cycle(self, cycle_id: str | None) -> None:
        with self._lock:
            self._current_cycle_id = cycle_id

    def set_decision(self, decision_id: int | None) -> None:
        with self._lock:
            self._current_decision_id = decision_id

    def compute_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Compute cost from token counts."""
        # Exact match
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            # Prefix match
            for k, v in MODEL_PRICING.items():
                if model.startswith(k) or k.startswith(model):
                    pricing = v
                    break
        if pricing is None:
            log.warning(f"Unknown model pricing: {model}")
            return 0.0
        inp, out = pricing
        return (input_tokens * inp + output_tokens * out) / 1_000_000

    def record(
        self,
        provider: str,
        model: str,
        role: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float | None = None,
        trade_id: Optional[int] = None,
    ) -> CostEntry:
        """Record a single API call. If cost_usd not given, compute from tokens."""
        if cost_usd is None:
            cost_usd = self.compute_cost(model, input_tokens, output_tokens)

        entry = CostEntry(
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model=model,
            role=role,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cycle_id=self._current_cycle_id,
            decision_id=self._current_decision_id,
            trade_id=trade_id,
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    def record_flat(
        self,
        provider: str,
        role: str,
        cost_usd: float,
        trade_id: Optional[int] = None,
    ) -> CostEntry:
        """Record a flat-fee call (e.g., Tavily search)."""
        return self.record(
            provider=provider,
            model=f"{provider}-{role}",
            role=role,
            cost_usd=cost_usd,
            trade_id=trade_id,
        )

    def cycle_total(self, cycle_id: str) -> float:
        with self._lock:
            return sum(e.cost_usd for e in self._entries if e.cycle_id == cycle_id)

    def decision_total(self, decision_id: int) -> float:
        with self._lock:
            return sum(e.cost_usd for e in self._entries if e.decision_id == decision_id)

    def breakdown_by_provider(self) -> dict[str, float]:
        with self._lock:
            result = defaultdict(float)
            for e in self._entries:
                result[e.provider] += e.cost_usd
            return dict(result)

    def breakdown_by_role(self, cycle_id: str | None = None) -> dict[str, float]:
        with self._lock:
            result = defaultdict(float)
            for e in self._entries:
                if cycle_id and e.cycle_id != cycle_id:
                    continue
                result[e.role] += e.cost_usd
            return dict(result)

    def total_today(self) -> float:
        today = datetime.now().date().isoformat()
        with self._lock:
            return sum(
                e.cost_usd for e in self._entries
                if e.timestamp.startswith(today)
            )


# Global singleton
_tracker: CostTracker | None = None


def get_tracker() -> CostTracker:
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker
