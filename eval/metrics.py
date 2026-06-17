"""Calibration metrics for binary forecasts — dependency-free (stdlib only)."""
from __future__ import annotations

import math


def brier(probs: list[float], outcomes: list[float]) -> float:
    """Mean squared error of predicted P(YES) vs outcome (0/1). Lower is better."""
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def log_loss(probs: list[float], outcomes: list[float], eps: float = 1e-9) -> float:
    total = 0.0
    for p, o in zip(probs, outcomes):
        p = min(max(p, eps), 1 - eps)
        total += -(o * math.log(p) + (1 - o) * math.log(1 - p))
    return total / len(probs)


def base_rate_brier(outcomes: list[float]) -> float:
    """Brier of the trivial 'always predict the base rate' baseline — the bar to beat."""
    if not outcomes:
        return 0.0
    base = sum(outcomes) / len(outcomes)
    return sum((base - o) ** 2 for o in outcomes) / len(outcomes)


def calibration_table(probs: list[float], outcomes: list[float], n_bins: int = 5) -> list[dict]:
    """Reliability bins: for each predicted-probability band, the realized YES rate."""
    table = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [
            i for i, p in enumerate(probs)
            if p >= lo and (p < hi or (b == n_bins - 1 and p <= hi))
        ]
        if not idx:
            table.append({"range": f"{lo:.1f}-{hi:.1f}", "n": 0, "avg_pred": None, "frac_yes": None})
            continue
        avg_pred = sum(probs[i] for i in idx) / len(idx)
        frac_yes = sum(outcomes[i] for i in idx) / len(idx)
        table.append({
            "range": f"{lo:.1f}-{hi:.1f}", "n": len(idx),
            "avg_pred": round(avg_pred, 3), "frac_yes": round(frac_yes, 3),
        })
    return table
