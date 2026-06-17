"""Database schema для детального tracking.

9 нових таблиць для root-cause analysis:
- cycle_metadata — ідентифікатор сканування
- trade_decisions — кожен проаналізований market (навіть skipped)
- llm_predictions — per-model response
- swarm_rounds — per-round aggregated stats
- swarm_agent_estimates — кожен агент в кожному раунді
- research_artifacts — Tavily/AI search records
- price_snapshots — trajectory за відкритими позиціями
- reeval_events — кожен reeval call
- error_log — будь-який error з повним контекстом
"""

from __future__ import annotations

import sqlite3
from llm_debate_swarm.utils.logger import get_logger

log = get_logger("tracking.schema")


SCHEMA_STATEMENTS = [
    # Cycle metadata
    """
    CREATE TABLE IF NOT EXISTS cycle_metadata (
        id TEXT PRIMARY KEY,
        start_ts TEXT NOT NULL,
        end_ts TEXT,
        markets_scanned INTEGER DEFAULT 0,
        markets_filtered INTEGER DEFAULT 0,
        markets_selected INTEGER DEFAULT 0,
        markets_researched INTEGER DEFAULT 0,
        signals_generated INTEGER DEFAULT 0,
        trades_executed INTEGER DEFAULT 0,
        trades_skipped INTEGER DEFAULT 0,
        total_cost_usd REAL DEFAULT 0.0,
        duration_sec REAL DEFAULT 0.0,
        error_count INTEGER DEFAULT 0,
        bot_version TEXT DEFAULT '',
        config_snapshot TEXT DEFAULT ''
    )
    """,

    # Trade decisions (per-market-per-cycle)
    """
    CREATE TABLE IF NOT EXISTS trade_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cycle_id TEXT,
        timestamp TEXT NOT NULL,
        market_id TEXT,
        market_question TEXT,
        market_category TEXT,
        market_pseudo_category TEXT,
        market_yes_price REAL,
        market_volume_24h REAL,
        market_liquidity REAL,
        days_to_resolution REAL,
        resolution_source TEXT,
        question_type TEXT,
        question_type_confidence REAL,
        edge_tier TEXT,
        stage_filter_passed INTEGER DEFAULT 0,
        stage_cooldown_passed INTEGER DEFAULT 0,
        stage_open_position_check INTEGER DEFAULT 0,
        stage_research_success INTEGER DEFAULT 0,
        stage_llm_success INTEGER DEFAULT 0,
        stage_swarm_success INTEGER DEFAULT 0,
        stage_edge_detected INTEGER DEFAULT 0,
        stage_risk_passed INTEGER DEFAULT 0,
        stage_order_placed INTEGER DEFAULT 0,
        terminated_at_stage TEXT,
        skip_reason TEXT,
        llm_consensus_prob REAL,
        llm_consensus_confidence REAL,
        llm_consensus_spread REAL,
        swarm_final_prob REAL,
        swarm_confidence REAL,
        swarm_anchor_shift REAL,
        swarm_convergence_ratio REAL,
        swarm_diversity_retry INTEGER DEFAULT 0,
        original_model_prob REAL,
        after_contrarian_prob REAL,
        after_extremize_prob REAL,
        diplomatic_correction_applied INTEGER DEFAULT 0,
        diplomatic_correction_delta REAL DEFAULT 0,
        sports_dampening_applied INTEGER DEFAULT 0,
        sports_dampening_delta REAL DEFAULT 0,
        commodity_correction_applied INTEGER DEFAULT 0,
        commodity_correction_delta REAL DEFAULT 0,
        final_model_prob REAL,
        edge_yes REAL,
        edge_no REAL,
        selected_side TEXT,
        final_edge REAL,
        signal_strength REAL,
        entry_guard_triggered INTEGER DEFAULT 0,
        trade_id INTEGER,
        api_cost_usd REAL DEFAULT 0.0,
        total_duration_sec REAL DEFAULT 0.0,
        FOREIGN KEY (cycle_id) REFERENCES cycle_metadata(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_decisions_cycle ON trade_decisions(cycle_id)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_market ON trade_decisions(market_id)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_trade ON trade_decisions(trade_id)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_ts ON trade_decisions(timestamp)",

    # LLM predictions
    """
    CREATE TABLE IF NOT EXISTS llm_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id INTEGER,
        timestamp TEXT,
        model_name TEXT,
        model_role TEXT,
        weight REAL,
        probability REAL,
        confidence REAL,
        reasoning TEXT,
        input_tokens INTEGER DEFAULT 0,
        output_tokens INTEGER DEFAULT 0,
        cost_usd REAL DEFAULT 0.0,
        latency_ms INTEGER DEFAULT 0,
        retry_count INTEGER DEFAULT 0,
        error TEXT,
        FOREIGN KEY (decision_id) REFERENCES trade_decisions(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_llm_decision ON llm_predictions(decision_id)",

    # Swarm rounds
    """
    CREATE TABLE IF NOT EXISTS swarm_rounds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id INTEGER,
        round_num INTEGER,
        round_type TEXT,
        agents_count INTEGER,
        mean_est REAL,
        median_est REAL,
        std_est REAL,
        min_est REAL,
        max_est REAL,
        unique_values_count INTEGER,
        parse_failures INTEGER DEFAULT 0,
        devils_advocates_count INTEGER DEFAULT 0,
        diversity_retry_triggered INTEGER DEFAULT 0,
        temperature_used REAL,
        duration_sec REAL,
        FOREIGN KEY (decision_id) REFERENCES trade_decisions(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_rounds_decision ON swarm_rounds(decision_id)",

    # Swarm agent estimates
    """
    CREATE TABLE IF NOT EXISTS swarm_agent_estimates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id INTEGER,
        round_num INTEGER,
        agent_id TEXT,
        agent_category TEXT,
        is_devils_advocate INTEGER DEFAULT 0,
        estimate REAL,
        confidence REAL,
        reasoning_short TEXT,
        key_factor TEXT,
        FOREIGN KEY (decision_id) REFERENCES trade_decisions(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_agents_decision ON swarm_agent_estimates(decision_id)",
    "CREATE INDEX IF NOT EXISTS idx_agents_round ON swarm_agent_estimates(decision_id, round_num)",

    # Research artifacts
    """
    CREATE TABLE IF NOT EXISTS research_artifacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id INTEGER,
        timestamp TEXT,
        search_query TEXT,
        query_type TEXT,
        results_count INTEGER DEFAULT 0,
        top_urls_json TEXT,
        ai_answer TEXT,
        raw_content_len INTEGER DEFAULT 0,
        cost_usd REAL DEFAULT 0.0,
        error TEXT,
        cached INTEGER DEFAULT 0,
        FOREIGN KEY (decision_id) REFERENCES trade_decisions(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_research_decision ON research_artifacts(decision_id)",

    # Price snapshots
    """
    CREATE TABLE IF NOT EXISTS price_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id INTEGER,
        timestamp TEXT,
        market_yes_price REAL,
        market_no_price REAL,
        our_side_price REAL,
        volume_24h REAL,
        liquidity REAL,
        unrealized_pnl REAL,
        unrealized_pnl_pct REAL,
        FOREIGN KEY (trade_id) REFERENCES trades(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_snapshots_trade ON price_snapshots(trade_id)",
    "CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON price_snapshots(timestamp)",

    # Reeval events
    """
    CREATE TABLE IF NOT EXISTS reeval_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id INTEGER,
        timestamp TEXT,
        trigger_type TEXT,
        trigger_detail TEXT,
        entry_yes_prob REAL,
        entry_price REAL,
        current_yes_market REAL,
        our_current_price REAL,
        unrealized_pnl_pct REAL,
        age_hours REAL,
        reeval_yes_prob REAL,
        reeval_reasoning TEXT,
        reeval_confidence TEXT,
        materially_changed INTEGER DEFAULT 0,
        drift REAL,
        market_disagreement REAL,
        hallucination_flag INTEGER DEFAULT 0,
        would_close INTEGER DEFAULT 0,
        guard_disagreement_triggered INTEGER DEFAULT 0,
        guard_direction_triggered INTEGER DEFAULT 0,
        final_action TEXT,
        news_found INTEGER DEFAULT 0,
        news_headlines_json TEXT,
        cost_usd REAL DEFAULT 0.0,
        FOREIGN KEY (trade_id) REFERENCES trades(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_reeval_trade ON reeval_events(trade_id)",
    "CREATE INDEX IF NOT EXISTS idx_reeval_ts ON reeval_events(timestamp)",

    # Error log
    """
    CREATE TABLE IF NOT EXISTS error_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        cycle_id TEXT,
        decision_id INTEGER,
        trade_id INTEGER,
        component TEXT,
        error_type TEXT,
        error_message TEXT,
        stack_trace TEXT,
        context_json TEXT,
        recoverable INTEGER DEFAULT 1,
        recovery_action TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_errors_cycle ON error_log(cycle_id)",
    "CREATE INDEX IF NOT EXISTS idx_errors_ts ON error_log(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_errors_component ON error_log(component)",
]


def apply_schema(db_path: str) -> None:
    """Create all new tracking tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for stmt in SCHEMA_STATEMENTS:
            try:
                cur.execute(stmt)
            except sqlite3.Error as e:
                log.warning(f"Schema stmt failed: {stmt[:80]}... — {e}")
        conn.commit()
        log.info(f"Schema applied: {len(SCHEMA_STATEMENTS)} statements")
    finally:
        conn.close()


def get_table_row_counts(db_path: str) -> dict[str, int]:
    """Return row count per tracking table for diagnostics."""
    tables = [
        "cycle_metadata", "trade_decisions", "llm_predictions",
        "swarm_rounds", "swarm_agent_estimates", "research_artifacts",
        "price_snapshots", "reeval_events", "error_log",
    ]
    conn = sqlite3.connect(db_path)
    counts = {}
    try:
        for t in tables:
            try:
                cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                counts[t] = cnt
            except sqlite3.Error:
                counts[t] = -1
    finally:
        conn.close()
    return counts
