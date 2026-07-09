"""Tests for the offline forecasting metrics (Brier skill, ECE, bootstrap CI, Wilson).

All deterministic, stdlib-only, no network — these are the guarantees the eval
harness and the CI regression gate rely on.
"""
from __future__ import annotations

from eval.metrics import (
    bootstrap_ci,
    brier,
    brier_skill_score,
    expected_calibration_error,
    wilson_interval,
)


def test_brier_skill_score_perfect_and_worse():
    # Perfect predictions -> Brier 0 -> skill 1.0 vs base rate.
    assert brier_skill_score([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1]) == 1.0
    # Confidently wrong -> worse than base rate -> negative skill.
    assert brier_skill_score([1.0, 1.0, 0.0, 0.0], [0, 0, 1, 1]) < 0.0


def test_ece_zero_when_perfectly_calibrated():
    assert expected_calibration_error([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1]) == 0.0
    # A systematically overconfident set has positive ECE.
    assert expected_calibration_error([0.9, 0.9, 0.9, 0.9], [1, 0, 0, 0]) > 0.0


def test_bootstrap_ci_is_deterministic_and_brackets_point():
    probs = [0.2, 0.4, 0.6, 0.8, 0.5, 0.5]
    outs = [0, 0, 1, 1, 0, 1]
    lo, hi = bootstrap_ci(brier, probs, outs, n_boot=500, seed=0)
    assert lo <= hi
    assert 0.0 <= lo and hi <= 1.0
    # Reproducible: same seed -> same interval.
    assert bootstrap_ci(brier, probs, outs, n_boot=500, seed=0) == (lo, hi)
    # A degenerate perfect set has a zero-width CI at 0.
    assert bootstrap_ci(brier, [0.0, 1.0], [0, 1], n_boot=200, seed=1) == (0.0, 0.0)


def test_wilson_interval_brackets_proportion():
    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi
    assert 0.0 < lo and hi < 1.0
