"""Calibration metrics for binary forecasts — dependency-free (stdlib only)."""
from __future__ import annotations

import math
import random


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


def brier_skill_score(
    probs: list[float], outcomes: list[float], baseline_brier: float | None = None
) -> float:
    """Skill vs a reference forecast (default: always-base-rate).

    1 = perfect, 0 = no better than the baseline, < 0 = worse. This is how you
    defend "the swarm beats the baseline" — the raw Brier alone doesn't say it.
    """
    ref = baseline_brier if baseline_brier is not None else base_rate_brier(outcomes)
    if ref <= 0:
        return 0.0
    return 1.0 - brier(probs, outcomes) / ref


def expected_calibration_error(
    probs: list[float], outcomes: list[float], n_bins: int = 10
) -> float:
    """ECE: sample-weighted mean gap between predicted probability and realized
    frequency across bins. 0 = perfectly calibrated."""
    n = len(probs)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [
            i for i, p in enumerate(probs)
            if p >= lo and (p < hi or (b == n_bins - 1 and p <= hi))
        ]
        if not idx:
            continue
        avg_pred = sum(probs[i] for i in idx) / len(idx)
        frac_yes = sum(outcomes[i] for i in idx) / len(idx)
        total += (len(idx) / n) * abs(avg_pred - frac_yes)
    return total


def bootstrap_ci(
    metric_fn,
    probs: list[float],
    outcomes: list[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for any metric(probs, outcomes).

    Deterministic given ``seed`` (so CI is reproducible and CI-gate-able). At the
    n≈50 scale of this eval the interval is honestly wide — that width is the
    point, not a defect.
    """
    n = len(probs)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(metric_fn([probs[i] for i in idx], [outcomes[i] for i in idx]))
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[int((1 - alpha / 2) * n_boot)]
    return (round(lo, 4), round(hi, 4))


def wilson_interval(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (base-rate uncertainty)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (round(center - half, 4), round(center + half, 4))
