"""Tests for the offline rigorous eval report — pure, deterministic, no network."""
from __future__ import annotations

from eval.analyze import build_report


def _rows():
    # 8 questions; consensus tracks the outcome fairly well, swarm is None
    # (consensus-only run) to exercise the honest "pending" path.
    data = [
        (0.1, 0), (0.2, 0), (0.3, 0), (0.4, 1),
        (0.6, 0), (0.7, 1), (0.8, 1), (0.9, 1),
    ]
    return [
        {"question": f"q{i}", "outcome": float(o), "consensus": p,
         "swarm": None, "combined": p}
        for i, (p, o) in enumerate(data)
    ]


def test_report_has_cis_and_skill_and_pending():
    report = build_report(_rows())

    assert report["n_questions"] == 8
    assert 0.0 <= report["base_rate"] <= 1.0
    lo, hi = report["base_rate_ci95_wilson"]
    assert lo < report["base_rate"] < hi

    cons = report["consensus"]
    assert cons["n"] == 8
    assert "brier_ci95" in cons and cons["brier_ci95"][0] <= cons["brier_ci95"][1]
    assert "ece" in cons
    assert "brier_skill_vs_base_rate" in cons
    assert len(cons["reliability_curve"]) == 10  # 10-bin reliability curve

    # swarm was None everywhere -> not scored, and honestly flagged as pending.
    assert "swarm" not in report
    joined = " ".join(report["pending"])
    assert "single_model_baseline" in joined
    assert "consensus-only" in joined


def test_market_price_baseline_when_supplied():
    rows = _rows()
    prices = {r["question"]: 0.5 for r in rows}
    report = build_report(rows, market_prices=prices)
    assert "market_price_baseline" in report
    assert report["market_price_baseline"]["n"] == 8
