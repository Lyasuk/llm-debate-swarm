"""Thin FastAPI serving layer — a *deploy proof*, not a production service.

- ``GET /health`` — liveness, touches no dependency.
- ``GET /ready`` — readiness, checks a provider key is configured.
- ``POST /forecast`` — one forecast through the primary LangGraph path, with a
  per-run timeout budget.

Batch/queue/worker/DLQ are deliberately **documented, not built** at this scale
(see ``DESIGN_DOC.md`` and the cut-list in ``PHASE2_HARDENING_SPEC.md``). Tracing
is wired from env at startup, so traces flow to Langfuse / LangSmith / any OTLP
collector without code changes.

Run: ``uvicorn llm_debate_swarm.service:app`` (needs the ``serve`` extra).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm_debate_swarm.config import get_optional_api_key
from llm_debate_swarm.graph import forecast_with_graph
from llm_debate_swarm.obs.tracing import setup_tracing

_PROVIDERS = ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")


def _has_provider() -> bool:
    return any(get_optional_api_key(k) for k in _PROVIDERS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_tracing("llm-debate-swarm-api")  # exports to whatever OTLP backend env configures
    yield


app = FastAPI(title="llm-debate-swarm", version="0.1.0", lifespan=lifespan)


class ForecastRequest(BaseModel):
    question: str = Field(min_length=3)
    category: str = ""
    prior: float | None = Field(default=None, ge=0.0, le=1.0)
    horizon_days: float | None = Field(default=None, ge=0.0)
    # Default OFF: the 30-agent swarm is ~200x the cost of consensus (see DESIGN_DOC §7).
    use_swarm: bool = False
    timeout_sec: float = Field(default=120.0, gt=0.0)


class ForecastResponse(BaseModel):
    question: str
    probability: float
    confidence: float
    disagreement: float
    consensus_probability: float | None
    swarm_probability: float | None
    question_type: str
    cost_usd: float


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict:
    if not _has_provider():
        raise HTTPException(status_code=503, detail="no provider API key configured")
    return {"status": "ready"}


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest) -> ForecastResponse:
    if not _has_provider():
        raise HTTPException(status_code=503, detail="no provider API key configured")
    try:
        v = await forecast_with_graph(
            req.question,
            category=req.category,
            prior=req.prior,
            horizon_days=req.horizon_days,
            use_swarm=req.use_swarm,
            timeout_sec=req.timeout_sec,
        )
    except TimeoutError:  # asyncio.TimeoutError is builtin TimeoutError on py3.11+
        raise HTTPException(status_code=504, detail="forecast timed out")
    except RuntimeError as exc:  # e.g. all providers failed -> no forecast produced
        raise HTTPException(status_code=502, detail=str(exc))
    return ForecastResponse(
        question=v.question,
        probability=v.probability,
        confidence=v.confidence,
        disagreement=v.disagreement,
        consensus_probability=v.consensus_probability,
        swarm_probability=v.swarm_probability,
        question_type=v.question_type,
        cost_usd=v.cost_usd,
    )
