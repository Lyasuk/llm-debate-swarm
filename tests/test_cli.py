"""CLI smoke tests — prove `forecast` runs end-to-end through both orchestrators.

A working model is mocked (no network); we assert the CLI wiring produces a
verdict. (With zero working models the engine correctly refuses to forecast —
that path is covered by the engine/graph tests, not here.)
"""
from __future__ import annotations

from click.testing import CliRunner

from llm_debate_swarm.analysis.multi_llm_analyzer import LLMPrediction, MultiLLMAnalyzer
from llm_debate_swarm.cli import cli


def _valid_pred(model_config):
    return LLMPrediction(
        model_name=model_config.name, provider=model_config.provider,
        probability=0.62, confidence="high", input_tokens=8, output_tokens=4,
    )


def _mock_working_model(monkeypatch):
    async def fake(self, model_config, prompt, *, question_id=None):
        return _valid_pred(model_config)

    monkeypatch.setattr(MultiLLMAnalyzer, "_query_model", fake)


def test_forecast_cli_runs_through_graph_by_default(monkeypatch):
    _mock_working_model(monkeypatch)
    result = CliRunner().invoke(cli, ["forecast", "Will it rain tomorrow?", "--no-swarm"])
    assert result.exit_code == 0, result.output
    assert "P(YES)" in result.output


def test_forecast_cli_async_engine_also_works(monkeypatch):
    _mock_working_model(monkeypatch)
    result = CliRunner().invoke(
        cli, ["forecast", "Will it rain tomorrow?", "--no-swarm", "--engine", "async"]
    )
    assert result.exit_code == 0, result.output
    assert "P(YES)" in result.output
