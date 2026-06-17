"""Statistical aggregation with debiasing corrections."""

from __future__ import annotations

import statistics


def trimmed_mean(values: list[float], trim_pct: float = 0.10) -> float:
    """Compute trimmed mean — remove top/bottom trim_pct before averaging."""
    if not values:
        return 0.5
    if len(values) < 4:
        return statistics.mean(values)

    sorted_vals = sorted(values)
    trim_count = max(1, int(len(sorted_vals) * trim_pct))
    trimmed = sorted_vals[trim_count:-trim_count] if trim_count > 0 else sorted_vals

    if not trimmed:
        return statistics.mean(values)

    return statistics.mean(trimmed)


def compute_confidence(values: list[float]) -> float:
    """Confidence based on agent agreement. Low spread = high confidence."""
    if len(values) < 2:
        return 0.5

    std = statistics.stdev(values)
    # std 0.05 -> 0.85, std 0.10 -> 0.70, std 0.20 -> 0.40, std 0.33 -> 0.01
    conf = 1.0 - std * 3.0
    return max(0.05, min(0.95, conf))


def detect_anchoring(
    blind_estimates: list[float], aware_estimates: list[float]
) -> float:
    """Measure anchoring: absolute shift from blind to price-aware estimates.

    Returns shift magnitude. >0.10 = significant anchoring.
    """
    if not blind_estimates or not aware_estimates:
        return 0.0

    blind_mean = statistics.mean(blind_estimates)
    aware_mean = statistics.mean(aware_estimates)
    return abs(blind_mean - aware_mean)


def convergence_ratio(r1_std: float, r7_std: float) -> float:
    """Ratio of final std to initial std. <0.4 = possible groupthink."""
    if r1_std <= 0.001:
        return 1.0  # no initial spread, can't converge
    return r7_std / r1_std


def premortem_impact(
    pre_estimates: list[float], post_estimates: list[float]
) -> float:
    """Fraction of agents who changed >5% after pre-mortem exercise."""
    if not pre_estimates or len(pre_estimates) != len(post_estimates):
        return 0.0

    changed = sum(
        1 for pre, post in zip(pre_estimates, post_estimates)
        if abs(pre - post) > 0.05
    )
    return changed / len(pre_estimates)


def adjusted_probability(
    raw_mean: float,
    blind_mean: float,
    aware_mean: float,
    anchoring_shift: float,
    convergence: float,
    premortem_frac: float,
) -> float:
    """Apply debiasing corrections to raw swarm aggregation.

    Corrections:
    - Anchoring: if shift > 0.10, weight blind mean 60% / aware 40%
    - Groupthink: if convergence < 0.40, pull 5% toward 50%
    - Pre-mortem: if >30% agents changed, pull 3% toward 50%
    """
    result = raw_mean

    # Anchoring correction: trust blind estimates more
    if anchoring_shift > 0.10 and blind_mean > 0 and aware_mean > 0:
        result = blind_mean * 0.60 + aware_mean * 0.40

    # Groupthink correction: pull toward 50%
    if convergence < 0.40:
        nudge = 0.05 * (1.0 if result > 0.5 else -1.0)
        result -= nudge

    # Pre-mortem impact correction: agents found real blind spots
    if premortem_frac > 0.30:
        nudge = 0.03 * (1.0 if result > 0.5 else -1.0)
        result -= nudge

    return max(0.01, min(0.99, result))


def extract_valid_estimates(
    estimates: list[float | None], min_valid: float = 0.01, max_valid: float = 0.99
) -> list[float]:
    """Filter and clamp estimates, removing None and invalid values."""
    result = []
    for est in estimates:
        if est is not None and isinstance(est, (int, float)):
            result.append(max(min_valid, min(max_valid, float(est))))
    return result
