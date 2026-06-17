"""Statistical meta aggregator — replaces LLM meta synthesis.

Motivation: Claude Sonnet meta anchors на extreme outliers
(D28: R7 mean=0.366, Claude meta=0.125 — 24pp gap due to premium
bucket's 0.02 outliers). Statistical methods are robust to outliers.

Algorithms:
- Trimmed mean 20% (excludes top/bottom 10% each)
- Weighted median by confidence
- MAD (Median Absolute Deviation) — robust spread
- Outlier rejection via MAD threshold
- Bimodal detector (std > 0.20)
- Bootstrap confidence interval

Output shape matches LLM meta для downstream compatibility.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Optional

from llm_debate_swarm.utils.logger import get_logger

log = get_logger("swarm.statistical_meta")


@dataclass
class StatisticalMetaResult:
    """Result of statistical aggregation."""
    probability: float
    confidence: float
    trimmed_mean: float
    median: float
    mad: float
    outliers_rejected: int
    is_bimodal: bool
    bimodal_low: Optional[float]
    bimodal_high: Optional[float]
    ci_low: float
    ci_high: float
    reasoning: str


def trimmed_mean(values: list[float], trim: float = 0.10) -> float:
    """Trim top/bottom trim fraction, return mean of remainder.

    Example: trim=0.10 → excludes 10% highest and 10% lowest.
    """
    if not values:
        return 0.5
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    k = int(n * trim)
    if k * 2 >= n:
        return statistics.mean(sorted_vals)
    trimmed = sorted_vals[k : n - k]
    return statistics.mean(trimmed)


def weighted_median(values: list[float], weights: list[float]) -> float:
    """Median weighted by weights. If weights sum to 0, uses unweighted median."""
    if not values:
        return 0.5
    if not weights or sum(weights) == 0:
        return statistics.median(values)

    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total_w = sum(weights)
    cumsum = 0.0
    for v, w in pairs:
        cumsum += w
        if cumsum >= total_w / 2:
            return v
    return pairs[-1][0]


def mad(values: list[float]) -> float:
    """Median Absolute Deviation — robust spread metric."""
    if not values:
        return 0.0
    med = statistics.median(values)
    abs_devs = [abs(v - med) for v in values]
    return statistics.median(abs_devs)


def reject_outliers(
    values: list[float],
    threshold_mad: float = 3.0,
) -> tuple[list[float], list[float]]:
    """Split values into core (within threshold × MAD) and outliers.

    Returns (core_values, outlier_values).
    """
    if len(values) < 3:
        return list(values), []

    med = statistics.median(values)
    m = mad(values)
    if m == 0:
        # All values equal to median — keep all
        return list(values), []

    core = [v for v in values if abs(v - med) <= threshold_mad * m]
    outliers = [v for v in values if abs(v - med) > threshold_mad * m]
    return core, outliers


def detect_bimodal(
    values: list[float],
    std_threshold: float = 0.20,
    gap_threshold: float = 0.20,
) -> tuple[bool, Optional[float], Optional[float]]:
    """Detect if distribution is bimodal.

    Returns (is_bimodal, low_mode_mean, high_mode_mean).

    Method: if std > threshold AND split at median gives two modes
    with gap > gap_threshold → bimodal.
    """
    if len(values) < 6:
        return False, None, None

    std = statistics.stdev(values)
    if std <= std_threshold:
        return False, None, None

    # Simple split at median
    med = statistics.median(values)
    low = [v for v in values if v < med]
    high = [v for v in values if v >= med]

    if len(low) < 3 or len(high) < 3:
        return False, None, None

    low_mode = statistics.mean(low)
    high_mode = statistics.mean(high)

    if high_mode - low_mode >= gap_threshold:
        return True, low_mode, high_mode
    return False, None, None


def bootstrap_ci(
    values: list[float],
    iterations: int = 1000,
    ci_level: float = 0.90,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    if len(values) < 2:
        v = values[0] if values else 0.5
        return v, v

    n = len(values)
    means = []
    for _ in range(iterations):
        sample = [random.choice(values) for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    low_idx = int(iterations * (1 - ci_level) / 2)
    high_idx = int(iterations * (1 - (1 - ci_level) / 2))
    return means[low_idx], means[high_idx - 1]


def aggregate(
    estimates: list[float],
    confidences: Optional[list[float]] = None,
    market_price: float = 0.5,
) -> StatisticalMetaResult:
    """Main entry point. Aggregate N agent estimates to final probability.

    Strategy:
    1. Reject outliers via MAD (3× threshold)
    2. Detect bimodal on core
    3. If bimodal: pick mode farther from market (contrarian signal)
    4. If unimodal: weighted combo of trimmed_mean + median
    5. Confidence from bootstrap CI width
    """
    if not estimates:
        return StatisticalMetaResult(
            probability=0.5, confidence=0.3,
            trimmed_mean=0.5, median=0.5, mad=0.0,
            outliers_rejected=0, is_bimodal=False,
            bimodal_low=None, bimodal_high=None,
            ci_low=0.0, ci_high=1.0,
            reasoning="No estimates provided",
        )

    # Step 1: reject outliers
    core, outliers = reject_outliers(estimates, threshold_mad=3.0)
    if len(core) < 3:
        # Too aggressive rejection, fall back to all
        core = list(estimates)
        outliers = []

    # Step 2: detect bimodal on core
    is_bimodal, low_mode, high_mode = detect_bimodal(core)

    # Step 3: compute stats
    tm = trimmed_mean(core, trim=0.10)
    med = statistics.median(core)
    m = mad(core)
    ci_low, ci_high = bootstrap_ci(core)

    # Step 4: decide final probability
    if is_bimodal and low_mode is not None and high_mode is not None:
        low_gap = abs(low_mode - market_price)
        high_gap = abs(high_mode - market_price)
        # Choose mode FARTHER from market (stronger signal)
        if high_gap > low_gap:
            final = high_mode
            reasoning = (
                f"Bimodal detected: low_mode={low_mode:.3f}, high_mode={high_mode:.3f}. "
                f"Chose high mode (gap from market={high_gap:.3f} vs low={low_gap:.3f})"
            )
        else:
            final = low_mode
            reasoning = (
                f"Bimodal detected: low_mode={low_mode:.3f}, high_mode={high_mode:.3f}. "
                f"Chose low mode (gap from market={low_gap:.3f} vs high={high_gap:.3f})"
            )
    else:
        # Unimodal: weighted combo
        final = 0.5 * tm + 0.5 * med
        reasoning = (
            f"Unimodal: trimmed_mean={tm:.3f}, median={med:.3f}, "
            f"outliers_rejected={len(outliers)}"
        )

    # Confidence from CI width — narrower CI = more confident
    ci_width = ci_high - ci_low
    confidence = max(0.30, min(0.95, 1.0 - ci_width * 1.5))

    return StatisticalMetaResult(
        probability=max(0.01, min(0.99, final)),
        confidence=confidence,
        trimmed_mean=tm,
        median=med,
        mad=m,
        outliers_rejected=len(outliers),
        is_bimodal=is_bimodal,
        bimodal_low=low_mode,
        bimodal_high=high_mode,
        ci_low=ci_low,
        ci_high=ci_high,
        reasoning=reasoning,
    )
