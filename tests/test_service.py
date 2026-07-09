"""FastAPI serving-layer tests — offline, a working model is mocked."""
from __future__ import annotations

from fastapi.testclient import TestClient

from llm_debate_swarm.analysis.multi_llm_analyzer import LLMPrediction, MultiLLMAnalyzer
from llm_debate_swarm.service import _PROVIDERS, app


def test_health_touches_no_dependency():
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_503_without_provider_key(monkeypatch):
    for k in _PROVIDERS:
        monkeypatch.delenv(k, raising=False)
    with TestClient(app) as c:
        assert c.get("/ready").status_code == 503


def test_forecast_ok_with_mocked_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    async def fake(self, model_config, prompt, *, question_id=None):
        return LLMPrediction(
            model_name=model_config.name, provider=model_config.provider,
            probability=0.61, confidence="high", input_tokens=5, output_tokens=3,
        )

    monkeypatch.setattr(MultiLLMAnalyzer, "_query_model", fake)
    with TestClient(app) as c:
        r = c.post("/forecast", json={"question": "Will it rain tomorrow?", "use_swarm": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["question_type"]


def test_forecast_503_without_provider_key(monkeypatch):
    for k in _PROVIDERS:
        monkeypatch.delenv(k, raising=False)
    with TestClient(app) as c:
        r = c.post("/forecast", json={"question": "Will it rain tomorrow?"})
    assert r.status_code == 503
