"""Configuration system for llm-debate-swarm (Pydantic models).

Slimmed, domain-neutral config: only the sections the debate/consensus engine
needs. The original exchange/trading sections were dropped during extraction;
``load_config`` also silently ignores any legacy sections so an old config file
won't break the engine.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SourceConfig(BaseModel):
    enabled: bool = False
    max_results: int = 10
    max_articles: int = 20
    subreddits: list[str] = Field(default_factory=list)


class ResearchConfig(BaseModel):
    sources: dict[str, SourceConfig] = Field(default_factory=dict)
    document_max_chars: int = 15000
    cache_ttl_hours: int = 6


class LLMModelConfig(BaseModel):
    name: str
    provider: str  # "anthropic" | "openai" | "google" | "groq"
    weight: float = 0.33


class AnalysisConfig(BaseModel):
    models: list[LLMModelConfig] = Field(default_factory=list)
    extremizing_factor: float = 1.15
    min_models_required: int = 1


class SwarmConfig(BaseModel):
    enabled: bool = True
    provider: str = "gemini"  # "gemini" | "groq" | "openai" | "gemini_multi"
    swarm_model: str = "gpt-4o-mini"
    meta_model: str = "gpt-4o"
    agent_count: int = 40
    debate_rounds: int = 7
    max_tokens_per_response: int = 800
    temperature: float = 0.7
    batch_size: int = 8
    timeout_sec: int = 180
    # Multi-model bucket mapping (used when provider == "gemini_multi")
    bucket_models: dict[str, str] = Field(default_factory=lambda: {
        "premium": "gemini-2.5-flash-lite",
        "standard": "gemma-3-27b-it",
        "nano": "gemma-3-4b-it",
    })
    # Per-model daily quotas for failover (requests per day)
    bucket_rpd_limits: dict[str, int] = Field(default_factory=lambda: {
        "premium": 500,
        "standard": 14400,
        "nano": 14400,
    })
    trim_pct: float = 0.10
    devils_advocate_count: int = 3
    blind_rounds: int = 3
    premortem_round: int = 6


class TrackingConfig(BaseModel):
    db_path: str = "data/runs.db"
    report_interval: str = "daily"


class AppConfig(BaseModel):
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)


_KNOWN_SECTIONS = {"research", "analysis", "swarm", "tracking"}


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from a YAML file (defaults to project-root config.yaml)."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"

    config_path = Path(config_path)
    data: dict[str, Any] = {}

    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    # Drop any legacy sections not in the slim schema (forward/backward compat).
    data = {k: v for k, v in data.items() if k in _KNOWN_SECTIONS}

    # Parse nested source configs (accept either dict or bare bool).
    if "research" in data and isinstance(data["research"], dict) and "sources" in data["research"]:
        raw_sources = data["research"]["sources"] or {}
        parsed: dict[str, dict[str, Any]] = {}
        for name, val in raw_sources.items():
            parsed[name] = val if isinstance(val, dict) else {"enabled": bool(val)}
        data["research"]["sources"] = parsed

    return AppConfig(**data)


def get_api_key(name: str) -> str:
    """Get an API key from the environment; raise if missing."""
    val = os.environ.get(name, "")
    if not val:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return val


def get_optional_api_key(name: str) -> str | None:
    """Get an API key from the environment; return None if missing."""
    return os.environ.get(name) or None
