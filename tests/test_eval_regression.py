"""Eval regression gate — runs in CI.

The committed calibration is a contract: if a prompt/model/graph change makes the
forecaster worse than a ceiling calibrated from the measured result, the build
goes red. The ceiling is derived from data (committed consensus Brier ≈ 0.223),
not a default — and we prove the gate actually bites.
"""
from __future__ import annotations

import json
import pathlib

from eval.metrics import brier

# Measured committed consensus Brier is 0.223; allow a small margin, trip on real
# regressions. Tighten this as n grows and the estimate stabilizes.
BRIER_CEILING = 0.24

_RESULTS = pathlib.Path(__file__).resolve().parent.parent / "eval" / "results" / "consensus.json"


def _committed_consensus():
    rows = json.loads(_RESULTS.read_text(encoding="utf-8"))["rows"]
    preds = [r["consensus"] for r in rows if r["consensus"] is not None]
    outs = [r["outcome"] for r in rows if r["consensus"] is not None]
    return preds, outs


def test_committed_consensus_brier_under_ceiling():
    preds, outs = _committed_consensus()
    assert len(preds) >= 25
    assert brier(preds, outs) <= BRIER_CEILING


def test_gate_bites_on_degraded_predictions():
    # Non-vacuous check: a deliberately-degraded forecaster (confidently wrong on
    # every question) must blow past the ceiling — proving the gate isn't a rubber stamp.
    _, outs = _committed_consensus()
    degraded = [1.0 - o for o in outs]
    assert brier(degraded, outs) > BRIER_CEILING
