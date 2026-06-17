"""Superforecaster techniques: reference class forecasting, extremizing, contrarian checks."""

from __future__ import annotations

import math

from llm_debate_swarm.utils.logger import get_logger

log = get_logger("analysis.superforecaster")


def extremize(probability: float, factor: float = 1.15) -> float:
    """Push probability away from 50% to compensate for conservatism.

    Superforecaster technique: aggregate predictions tend to be too
    conservative (cluster around 50%). Extremizing pushes them outward.

    Uses the log-odds extremizing formula:
        log_odds_new = factor * log_odds_original

    Args:
        probability: Original probability (0-1)
        factor: Extremizing factor (>1 = more extreme, <1 = less)

    Returns:
        Extremized probability
    """
    if probability <= 0.01 or probability >= 0.99:
        return probability

    # Convert to log-odds
    log_odds = math.log(probability / (1 - probability))

    # Extremize
    new_log_odds = log_odds * factor

    # Convert back to probability
    extremized = 1 / (1 + math.exp(-new_log_odds))

    # Clamp
    return max(0.01, min(0.99, extremized))


def contrarian_adjustment(
    predictions: list[float],
    base_uncertainty: float = 0.05,
) -> float:
    """Increase uncertainty when all models agree too strongly.

    If all LLMs say 80%+, it might mean they share a blind spot.
    This adds a small contrarian nudge toward 50%.

    Args:
        predictions: List of probability estimates from different sources
        base_uncertainty: Base uncertainty to add when sources agree

    Returns:
        Adjustment factor to add to probability (positive or negative)
    """
    if len(predictions) < 2:
        return 0.0

    avg = sum(predictions) / len(predictions)
    spread = max(predictions) - min(predictions)

    # If all sources agree (spread < 0.05) and probability is extreme
    if spread < 0.05 and abs(avg - 0.5) > 0.2:
        # Nudge toward 50%
        direction = -1 if avg > 0.5 else 1
        adjustment = direction * base_uncertainty * (1 - spread / 0.05)
        log.info(
            f"Contrarian adjustment: {adjustment:+.3f} "
            f"(all {len(predictions)} sources agree at ~{avg:.1%})"
        )
        return adjustment

    return 0.0


def reference_class_probability(
    question_keywords: list[str],
) -> float | None:
    """Rough base rate from reference class (hardcoded for common categories).

    This is a simplified version. In production, you'd query Metaculus
    or historical Polymarket data for similar resolved questions.

    Returns None if no reference class matches.
    """
    # Very rough base rates by category
    reference_classes: dict[str, float] = {
        "reelection": 0.55,    # Incumbents win ~55% of the time
        "impeach": 0.15,       # Impeachment attempts rarely succeed
        "war": 0.20,           # Major military conflicts are rare
        "recession": 0.30,     # Recessions happen ~30% of years
        "rate cut": 0.50,      # Fed rate decisions are roughly even
        "bitcoin": 0.50,       # Crypto predictions are ~coin flip
        "regulation": 0.40,    # New regulations pass ~40% of the time
        "merger": 0.65,        # Announced mergers complete ~65%
        "ipo": 0.50,           # IPO predictions are uncertain
        "default": 0.10,       # Sovereign defaults are rare
    }

    q_lower = " ".join(kw.lower() for kw in question_keywords)

    for keyword, base_rate in reference_classes.items():
        if keyword in q_lower:
            log.info(f"Reference class '{keyword}': base rate = {base_rate:.1%}")
            return base_rate

    return None
