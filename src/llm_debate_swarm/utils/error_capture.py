"""Error capture utilities — log exceptions з повним контекстом в error_log table.

Use cases:
1. @capture_errors decorator — ловить і логує exceptions від функцій
2. log_error() — manual logging з контекстом
3. ErrorContext — context manager для blocks
"""

from __future__ import annotations

import functools
import json
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from llm_debate_swarm.utils.logger import get_logger

log = get_logger("error_capture")

_DB_PATH: Optional[Path] = None


def set_db_path(db_path: str) -> None:
    """Set DB path for error logging (called once at startup)."""
    global _DB_PATH
    _DB_PATH = Path(db_path)


def _classify_error(exc: Exception) -> str:
    """Classify exception into error_type tag."""
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    if "429" in exc_str or "rate limit" in exc_str or "quota" in exc_str:
        return "rate_limit"
    if "529" in exc_str or "overloaded" in exc_str:
        return "api_overloaded"
    if "timeout" in exc_str or "timeouterror" in exc_type:
        return "api_timeout"
    if "403" in exc_str or "forbidden" in exc_str or "geo" in exc_str:
        return "geo_block"
    if "401" in exc_str or "unauthorized" in exc_str:
        return "auth_error"
    if "parse" in exc_str or "json" in exc_str and "decode" in exc_str:
        return "parse_fail"
    if "connection" in exc_str or "network" in exc_str:
        return "network_error"
    if "not found" in exc_str or "404" in exc_str:
        return "not_found"
    return "unknown"


def log_error(
    component: str,
    exc: Exception | None = None,
    error_type: str | None = None,
    message: str | None = None,
    cycle_id: str | None = None,
    decision_id: int | None = None,
    trade_id: int | None = None,
    context: dict | None = None,
    recoverable: bool = True,
    recovery_action: str = "",
) -> None:
    """Log error to error_log table.

    Don't raise on failure — logging errors must never break pipeline.
    """
    if _DB_PATH is None:
        return

    err_type = error_type or (_classify_error(exc) if exc else "manual")
    err_msg = message or (str(exc) if exc else "")
    stack = traceback.format_exc() if exc else ""
    ctx_json = ""
    if context:
        try:
            ctx_json = json.dumps(context, default=str)[:5000]
        except Exception:
            ctx_json = str(context)[:5000]

    try:
        conn = sqlite3.connect(_DB_PATH, timeout=5)
        conn.execute(
            """
            INSERT INTO error_log
            (timestamp, cycle_id, decision_id, trade_id, component,
             error_type, error_message, stack_trace, context_json,
             recoverable, recovery_action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                cycle_id,
                decision_id,
                trade_id,
                component,
                err_type,
                err_msg[:2000],
                stack[:5000],
                ctx_json,
                1 if recoverable else 0,
                recovery_action,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as log_exc:
        # Never raise from logging itself
        log.warning(f"error_log write failed: {log_exc}")


def capture_errors(component: str, reraise: bool = True):
    """Decorator that catches exceptions and logs them before optionally re-raising."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                log_error(component=component, exc=exc)
                if reraise:
                    raise
                return None

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                log_error(component=component, exc=exc)
                if reraise:
                    raise
                return None

        import asyncio
        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator


class ErrorContext:
    """Context manager для error capture в code blocks.

    with ErrorContext("research", cycle_id=cid, decision_id=did):
        # risky code
    """

    def __init__(
        self,
        component: str,
        cycle_id: str | None = None,
        decision_id: int | None = None,
        trade_id: int | None = None,
        reraise: bool = True,
        context: dict | None = None,
    ):
        self.component = component
        self.cycle_id = cycle_id
        self.decision_id = decision_id
        self.trade_id = trade_id
        self.reraise = reraise
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            log_error(
                component=self.component,
                exc=exc_val,
                cycle_id=self.cycle_id,
                decision_id=self.decision_id,
                trade_id=self.trade_id,
                context=self.context,
            )
            if not self.reraise:
                return True  # suppress
        return False
