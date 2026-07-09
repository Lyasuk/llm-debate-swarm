"""Offline, rigorous analysis of a committed eval results file.

Turns a run's per-question rows (produced by ``run_eval.py``) into a report a
reviewer can trust: Brier + bootstrap 95% CI, log-loss + CI, ECE, a reliability
curve, and a Brier skill score vs baselines (always-base-rate, and optionally
the market-implied price). **No API calls** — everything is recomputed from
already-committed predictions, so it is fully reproducible and can gate CI.

The headline claim is a *skill vs a baseline with a CI*, not a bare Brier number.

Baselines that need fresh LLM calls (a single-model no-debate forecaster, and a
per-model breakdown) are reported as ``pending`` — honestly, not faked.

Usage (from repo root)::

    python -m eval.analyze --results eval/results/consensus.json
    python -m eval.analyze --results eval/results/consensus.json \
        --market-prices eval/results/market_prices.json --out eval/results/analysis.json
"""
from __future__ import annotations

import argparse
import json

from eval.metrics import (
    base_rate_brier,
    bootstrap_ci,
    brier,
    brier_skill_score,
    calibration_table,
    expected_calibration_error,
    log_loss,
    wilson_interval,
)

_COLUMNS = ("consensus", "swarm", "combined")


def analyze_column(preds: list[float], outs: list[float], *, seed: int = 0) -> dict:
    """Full metric set for one predictor, every point estimate paired with a CI."""
    base = base_rate_brier(outs)
    return {
        "n": len(preds),
        "brier": round(brier(preds, outs), 4),
        "brier_ci95": bootstrap_ci(brier, preds, outs, seed=seed),
        "log_loss": round(log_loss(preds, outs), 4),
        "log_loss_ci95": bootstrap_ci(log_loss, preds, outs, seed=seed),
        "ece": round(expected_calibration_error(preds, outs), 4),
        "base_rate_brier": round(base, 4),
        "brier_skill_vs_base_rate": round(brier_skill_score(preds, outs, base), 4),
        "reliability_curve": calibration_table(preds, outs, n_bins=10),
    }


def build_report(rows: list[dict], market_prices: dict[str, float] | None = None) -> dict:
    """Assemble the rigorous report from committed per-question rows.

    ``rows`` items look like ``{question, outcome, consensus, swarm, combined, ...}``
    (any predictor may be ``None`` when that config was not run).
    """
    outs_all = [float(r["outcome"]) for r in rows]
    n = len(outs_all)
    report: dict = {
        "n_questions": n,
        "base_rate": round(sum(outs_all) / n, 4) if n else 0.0,
        "base_rate_ci95_wilson": wilson_interval(sum(outs_all), n),
    }

    for col in _COLUMNS:
        preds = [r[col] for r in rows if r.get(col) is not None]
        outs = [float(r["outcome"]) for r in rows if r.get(col) is not None]
        if len(preds) >= 5:
            report[col] = analyze_column(preds, outs)

    # Market-implied price baseline — the strong baseline a forecaster must beat.
    # Free to obtain (public Polymarket data), no LLM key; supplied via --market-prices.
    if market_prices:
        mp, mo = [], []
        for r in rows:
            price = market_prices.get(r["question"])
            if price is not None:
                mp.append(float(price))
                mo.append(float(r["outcome"]))
        if len(mp) >= 5:
            report["market_price_baseline"] = analyze_column(mp, mo)
    else:
        report.setdefault("pending", []).append(
            "market_price_baseline (a FAIR one needs the market's point-in-time price at "
            "forecast time; a resolved market's final price ≈ the outcome, so it is not a "
            "valid baseline — needs Polymarket price history; tracked as an ADR decision)"
        )

    # Honestly flag what still needs fresh LLM calls rather than faking it.
    report.setdefault("pending", []).extend([
        "single_model_baseline @temp0 (needs LLM keys)",
        "per_model_breakdown (needs LLM keys)",
    ])
    if not any(r.get("swarm") is not None for r in rows):
        report["pending"].append("swarm/combined calibration (consensus-only run committed)")
    return report


def _print_summary(report: dict) -> None:
    print(f"\n=== RIGOROUS ANALYSIS (n={report['n_questions']}) ===")
    br = report.get("base_rate_ci95_wilson")
    print(f"base rate = {report['base_rate']}  (95% Wilson {br})")
    for col in (*_COLUMNS, "market_price_baseline"):
        m = report.get(col)
        if not m:
            continue
        print(
            f"{col:22s} n={m['n']:3d}  Brier={m['brier']:.4f} "
            f"CI95{m['brier_ci95']}  ECE={m['ece']:.4f}  "
            f"skill_vs_base={m['brier_skill_vs_base_rate']:+.4f}"
        )
    if report.get("pending"):
        print("pending (honest):")
        for p in report["pending"]:
            print(f"  - {p}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="eval/results/consensus.json")
    p.add_argument("--market-prices", default="", help="Optional JSON {question: price}.")
    p.add_argument("--out", default="eval/results/analysis.json")
    args = p.parse_args()

    with open(args.results, encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    market_prices = None
    if args.market_prices:
        with open(args.market_prices, encoding="utf-8") as f:
            market_prices = json.load(f)

    report = build_report(rows, market_prices)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _print_summary(report)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
