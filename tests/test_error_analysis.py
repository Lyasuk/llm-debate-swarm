"""Tests for offline error-analysis bucketing — deterministic, no network."""
from __future__ import annotations

from eval.error_analysis import analyze_errors


def test_buckets_and_dominant_failure():
    rows = [
        # resolved YES but forecast low -> missed_yes (x3)
        {"question": "a", "outcome": 1.0, "consensus": 0.2},
        {"question": "b", "outcome": 1.0, "consensus": 0.3},
        {"question": "c", "outcome": 1.0, "consensus": 0.35},
        # resolved NO but forecast high -> missed_no (x1); most confident wrong call
        {"question": "d", "outcome": 0.0, "consensus": 0.9},
        # hedged wrong: NO but 0.55 -> wrong side, mild
        {"question": "e", "outcome": 0.0, "consensus": 0.55},
        # correct side
        {"question": "f", "outcome": 1.0, "consensus": 0.7},
        {"question": "g", "outcome": 0.0, "consensus": 0.2},
    ]
    a = analyze_errors(rows, worst_k=3)

    assert a["n"] == 7
    assert a["bucket_counts"]["missed_yes"] == 3
    assert a["bucket_counts"]["missed_no"] == 1
    assert a["bucket_counts"]["hedged_wrong"] == 1
    assert a["dominant_failure"] == "missed_yes"
    # worst call is the most confident wrong one (NO @ 0.8).
    assert a["worst"][0]["question"] == "d"
    assert len(a["worst"]) == 3
    # under-forecasting YES shows up as a lower mean forecast on YES than the truth.
    assert a["avg_pred_on_yes"] < 0.6
