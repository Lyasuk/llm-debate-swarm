"""DecisionLogger — writes rows to trade_decisions, cycle_metadata, and related tables.

Separated from TradeLogger so the existing trade-logging logic stays untouched
(tests/schemas) while we build a rich analytical layer on top.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_debate_swarm.utils.logger import get_logger

log = get_logger("tracking.decision")


class DecisionLogger:
    """Writes cycle, decision, and sub-entity rows defensively."""

    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10)

    # ------------------------------------------------------------------
    # Cycle metadata
    # ------------------------------------------------------------------

    def start_cycle(
        self,
        cycle_id: str,
        bot_version: str = "",
        config_snapshot: str = "",
    ) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cycle_metadata
                    (id, start_ts, bot_version, config_snapshot)
                    VALUES (?, ?, ?, ?)
                    """,
                    (cycle_id, datetime.now().isoformat(), bot_version, config_snapshot),
                )
        except Exception as exc:
            log.warning(f"start_cycle failed: {exc}")

    def finish_cycle(
        self,
        cycle_id: str,
        stats: dict,
    ) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE cycle_metadata SET
                        end_ts = ?,
                        markets_scanned = ?,
                        markets_filtered = ?,
                        markets_selected = ?,
                        markets_researched = ?,
                        signals_generated = ?,
                        trades_executed = ?,
                        trades_skipped = ?,
                        total_cost_usd = ?,
                        duration_sec = ?,
                        error_count = ?
                    WHERE id = ?
                    """,
                    (
                        datetime.now().isoformat(),
                        stats.get("markets_scanned", 0),
                        stats.get("markets_filtered", 0),
                        stats.get("markets_selected", 0),
                        stats.get("markets_researched", 0),
                        stats.get("signals_generated", 0),
                        stats.get("trades_executed", 0),
                        stats.get("trades_skipped", 0),
                        stats.get("total_cost_usd", 0.0),
                        stats.get("duration_sec", 0.0),
                        stats.get("error_count", 0),
                        cycle_id,
                    ),
                )
        except Exception as exc:
            log.warning(f"finish_cycle failed: {exc}")

    # ------------------------------------------------------------------
    # Trade decisions
    # ------------------------------------------------------------------

    def create_decision(self, cycle_id: str, market: Any) -> int | None:
        """Insert new decision row at start of market processing. Returns decision_id."""
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO trade_decisions
                    (cycle_id, timestamp, market_id, market_question, market_category,
                     market_yes_price, market_volume_24h, market_liquidity,
                     days_to_resolution, resolution_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cycle_id,
                        datetime.now().isoformat(),
                        getattr(market, "id", ""),
                        getattr(market, "question", "")[:500],
                        getattr(market, "category", "") or "",
                        getattr(market, "yes_price", 0.0),
                        getattr(market, "volume_24h", 0.0),
                        getattr(market, "liquidity", 0.0),
                        getattr(market, "days_to_resolution", 0.0),
                        (getattr(market, "resolution_source", "") or "")[:500],
                    ),
                )
                return cur.lastrowid
        except Exception as exc:
            log.warning(f"create_decision failed: {exc}")
            return None

    def update_decision(self, decision_id: int, **fields) -> None:
        """Update arbitrary fields on decision row."""
        if decision_id is None or not fields:
            return
        try:
            set_clause = ", ".join(f"{k} = ?" for k in fields.keys())
            values = list(fields.values()) + [decision_id]
            with self._connect() as conn:
                conn.execute(
                    f"UPDATE trade_decisions SET {set_clause} WHERE id = ?",
                    values,
                )
        except Exception as exc:
            log.warning(f"update_decision failed: {exc}")

    # ------------------------------------------------------------------
    # LLM predictions
    # ------------------------------------------------------------------

    def log_llm_prediction(
        self,
        decision_id: int | None,
        model_name: str,
        role: str,
        probability: float,
        confidence: float,
        reasoning: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: int = 0,
        weight: float = 1.0,
        retry_count: int = 0,
        error: str = "",
    ) -> None:
        if decision_id is None:
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO llm_predictions
                    (decision_id, timestamp, model_name, model_role, weight,
                     probability, confidence, reasoning, input_tokens, output_tokens,
                     cost_usd, latency_ms, retry_count, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        decision_id, datetime.now().isoformat(), model_name, role,
                        weight, probability, confidence, reasoning[:5000],
                        input_tokens, output_tokens, cost_usd, latency_ms,
                        retry_count, error[:500],
                    ),
                )
        except Exception as exc:
            log.warning(f"log_llm_prediction failed: {exc}")

    # ------------------------------------------------------------------
    # Swarm rounds + agents
    # ------------------------------------------------------------------

    def log_swarm_round(
        self,
        decision_id: int | None,
        round_num: int,
        round_type: str,
        stats: dict,
    ) -> None:
        if decision_id is None:
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO swarm_rounds
                    (decision_id, round_num, round_type, agents_count,
                     mean_est, median_est, std_est, min_est, max_est,
                     unique_values_count, parse_failures, devils_advocates_count,
                     diversity_retry_triggered, temperature_used, duration_sec)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        decision_id, round_num, round_type,
                        stats.get("agents_count", 0),
                        stats.get("mean", 0.0),
                        stats.get("median", 0.0),
                        stats.get("std", 0.0),
                        stats.get("min", 0.0),
                        stats.get("max", 0.0),
                        stats.get("unique", 0),
                        stats.get("parse_failures", 0),
                        stats.get("devils_advocates", 0),
                        1 if stats.get("diversity_retry") else 0,
                        stats.get("temperature", 0.0),
                        stats.get("duration_sec", 0.0),
                    ),
                )
        except Exception as exc:
            log.warning(f"log_swarm_round failed: {exc}")

    def log_swarm_agents_batch(
        self,
        decision_id: int | None,
        round_num: int,
        agents_data: list[dict],
    ) -> None:
        """Batch insert всіх агентів за раунд."""
        if decision_id is None or not agents_data:
            return
        try:
            rows = [
                (
                    decision_id, round_num,
                    a.get("agent_id", ""),
                    a.get("category", ""),
                    1 if a.get("is_devils_advocate") else 0,
                    a.get("estimate", 0.0),
                    a.get("confidence", 0.0),
                    (a.get("reasoning", "") or "")[:500],
                    (a.get("key_factor", "") or "")[:200],
                )
                for a in agents_data
            ]
            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT INTO swarm_agent_estimates
                    (decision_id, round_num, agent_id, agent_category,
                     is_devils_advocate, estimate, confidence,
                     reasoning_short, key_factor)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        except Exception as exc:
            log.warning(f"log_swarm_agents_batch failed: {exc}")

    # ------------------------------------------------------------------
    # Research artifacts
    # ------------------------------------------------------------------

    def log_research(
        self,
        decision_id: int | None,
        search_query: str,
        query_type: str = "primary",
        results_count: int = 0,
        top_urls: list[dict] | None = None,
        ai_answer: str = "",
        raw_content_len: int = 0,
        cost_usd: float = 0.0,
        error: str = "",
        cached: bool = False,
    ) -> None:
        if decision_id is None:
            return
        try:
            urls_json = json.dumps(top_urls or [], default=str)[:5000]
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO research_artifacts
                    (decision_id, timestamp, search_query, query_type, results_count,
                     top_urls_json, ai_answer, raw_content_len, cost_usd, error, cached)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        decision_id, datetime.now().isoformat(),
                        search_query[:500], query_type, results_count,
                        urls_json, ai_answer[:2000], raw_content_len,
                        cost_usd, error[:500], 1 if cached else 0,
                    ),
                )
        except Exception as exc:
            log.warning(f"log_research failed: {exc}")

    # ------------------------------------------------------------------
    # Price snapshots
    # ------------------------------------------------------------------

    def log_price_snapshot(
        self,
        trade_id: int,
        market_yes_price: float,
        volume_24h: float = 0.0,
        liquidity: float = 0.0,
        our_side_price: float | None = None,
        unrealized_pnl: float | None = None,
        unrealized_pnl_pct: float | None = None,
    ) -> None:
        try:
            market_no_price = 1.0 - market_yes_price
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO price_snapshots
                    (trade_id, timestamp, market_yes_price, market_no_price,
                     our_side_price, volume_24h, liquidity,
                     unrealized_pnl, unrealized_pnl_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade_id, datetime.now().isoformat(),
                        market_yes_price, market_no_price,
                        our_side_price, volume_24h, liquidity,
                        unrealized_pnl, unrealized_pnl_pct,
                    ),
                )
        except Exception as exc:
            log.warning(f"log_price_snapshot failed: {exc}")

    # ------------------------------------------------------------------
    # Reeval events
    # ------------------------------------------------------------------

    def log_reeval(
        self,
        trade_id: int,
        data: dict,
    ) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO reeval_events
                    (trade_id, timestamp, trigger_type, trigger_detail,
                     entry_yes_prob, entry_price, current_yes_market, our_current_price,
                     unrealized_pnl_pct, age_hours, reeval_yes_prob, reeval_reasoning,
                     reeval_confidence, materially_changed, drift, market_disagreement,
                     hallucination_flag, would_close, guard_disagreement_triggered,
                     guard_direction_triggered, final_action, news_found,
                     news_headlines_json, cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade_id, datetime.now().isoformat(),
                        data.get("trigger_type", ""),
                        data.get("trigger_detail", "")[:500],
                        data.get("entry_yes_prob"),
                        data.get("entry_price"),
                        data.get("current_yes_market"),
                        data.get("our_current_price"),
                        data.get("unrealized_pnl_pct"),
                        data.get("age_hours"),
                        data.get("reeval_yes_prob"),
                        (data.get("reeval_reasoning", "") or "")[:2000],
                        data.get("reeval_confidence", "low"),
                        1 if data.get("materially_changed") else 0,
                        data.get("drift"),
                        data.get("market_disagreement"),
                        1 if data.get("hallucination_flag") else 0,
                        1 if data.get("would_close") else 0,
                        1 if data.get("guard_disagreement_triggered") else 0,
                        1 if data.get("guard_direction_triggered") else 0,
                        data.get("final_action", ""),
                        1 if data.get("news_found") else 0,
                        json.dumps(data.get("news_headlines", []))[:2000],
                        data.get("cost_usd", 0.0),
                    ),
                )
        except Exception as exc:
            log.warning(f"log_reeval failed: {exc}")


_logger_instance: DecisionLogger | None = None


def get_logger_instance(db_path: str = "data/trades.db") -> DecisionLogger:
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = DecisionLogger(db_path)
    return _logger_instance
