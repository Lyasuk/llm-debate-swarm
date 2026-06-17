"""Structured logging setup for Poly Think."""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True, force_terminal=True, legacy_windows=False)


def setup_logger(name: str = "llm_debate_swarm", level: int = logging.INFO) -> logging.Logger:
    """Create a structured logger with rich output."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = RichHandler(
            console=_console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


def get_logger(module: str) -> logging.Logger:
    """Get a child logger for a specific module."""
    return logging.getLogger(f"llm_debate_swarm.{module}")
