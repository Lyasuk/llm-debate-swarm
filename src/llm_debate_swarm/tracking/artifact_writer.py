"""JSON artifact writer для повного контексту кожного decision.

Зберігає raw data який занадто великий або неструктурований для SQLite:
- Повні research documents
- Повні LLM prompts/responses
- Raw swarm responses per agent per round
- Повний edge computation breakdown

Структура:
  data/artifacts/YYYY-MM-DD/cycle_{uuid}/
    cycle_summary.json
    market_{id}_decision.json
    market_{id}_swarm.json
    reeval_{trade_id}_{timestamp}.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_debate_swarm.utils.logger import get_logger

log = get_logger("tracking.artifacts")


class ArtifactWriter:
    """Writes JSON artifacts with full pipeline context."""

    def __init__(self, base_dir: str = "data/artifacts"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _cycle_dir(self, cycle_id: str, timestamp: str | None = None) -> Path:
        """Returns path data/artifacts/YYYY-MM-DD/cycle_{uuid}/"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        date_str = timestamp[:10]  # YYYY-MM-DD
        d = self.base_dir / date_str / f"cycle_{cycle_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _safe_write(self, path: Path, data: dict) -> bool:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            return True
        except Exception as exc:
            log.warning(f"Artifact write failed {path}: {exc}")
            return False

    def write_cycle_summary(self, cycle_id: str, summary: dict) -> None:
        """Save cycle-level summary (configuration, counts, errors)."""
        d = self._cycle_dir(cycle_id, summary.get("start_ts"))
        self._safe_write(d / "cycle_summary.json", summary)

    def write_decision_artifact(
        self,
        cycle_id: str,
        market_id: str,
        decision_artifact: dict,
    ) -> None:
        """Save full decision pipeline context.

        decision_artifact structure:
          {
            "market": {...},
            "classification": {...},
            "research": {"queries": [...], "full_document": "...", "raw_results": [...]},
            "llm_prompt": "...",
            "llm_responses": [{"model": "...", "raw": "...", "parsed": {...}}],
            "swarm": {"rounds": [...], "meta": {...}},
            "edge_computation": {...},
            "position_sizing": {...},
            "decision": "TRADE"|"SKIP",
            "skip_reason": "..."
          }
        """
        d = self._cycle_dir(cycle_id, decision_artifact.get("timestamp"))
        safe_id = str(market_id).replace("/", "_")[:50]
        self._safe_write(d / f"market_{safe_id}_decision.json", decision_artifact)

    def write_swarm_artifact(
        self,
        cycle_id: str,
        market_id: str,
        swarm_artifact: dict,
    ) -> None:
        """Save full swarm debate context (separate file — може бути великий).

        swarm_artifact structure:
          {
            "rounds": [
              {
                "round": 1,
                "type": "BLIND",
                "prompt_sent": "...",
                "agents": [
                  {"persona_id": "x", "category": "Skeptic", "raw_response": "...",
                   "parsed_estimate": 0.34, "is_devils_advocate": false, "reasoning": "..."}
                ],
                "stats": {"mean": 0.35, "std": 0.06, "unique": 15}
              }
            ],
            "meta_synthesis": {
              "prompt": "...",
              "response": "...",
              "final_probability": 0.32
            },
            "cost_breakdown": {...}
          }
        """
        d = self._cycle_dir(cycle_id)
        safe_id = str(market_id).replace("/", "_")[:50]
        self._safe_write(d / f"market_{safe_id}_swarm.json", swarm_artifact)

    def write_reeval_artifact(
        self,
        trade_id: int,
        timestamp: str,
        reeval_artifact: dict,
    ) -> None:
        """Save full reeval context for a single reeval event.

        reeval_artifact structure:
          {
            "trade_id": N,
            "trigger": {...},
            "prompt_sent": "...",
            "llm_response": "...",
            "parsed": {...},
            "decision_flow": {...},
            "news_context": {"headlines": [...], "full_text": "..."}
          }
        """
        # Organize by date, not cycle (reevals are per-trade, cross-cycle)
        date_str = timestamp[:10]
        d = self.base_dir / date_str / "reevals"
        d.mkdir(parents=True, exist_ok=True)
        ts_safe = timestamp.replace(":", "-").replace(".", "-")[:19]
        self._safe_write(d / f"trade_{trade_id}_{ts_safe}.json", reeval_artifact)

    def write_error_artifact(
        self,
        cycle_id: str | None,
        error_data: dict,
    ) -> None:
        """Save detailed error context."""
        ts = error_data.get("timestamp", datetime.now().isoformat())
        date_str = ts[:10]
        d = self.base_dir / date_str / "errors"
        d.mkdir(parents=True, exist_ok=True)
        component = error_data.get("component", "unknown")
        ts_safe = ts.replace(":", "-").replace(".", "-")[:19]
        self._safe_write(d / f"{component}_{ts_safe}.json", error_data)


# Singleton pattern — use global instance
_instance: ArtifactWriter | None = None


def get_writer() -> ArtifactWriter:
    global _instance
    if _instance is None:
        _instance = ArtifactWriter()
    return _instance
