# Postmortem — the forecaster is under-confident (compressed calibration)

**Date:** 2026-07-09 · **Status:** action items open · **Severity:** quality (not an outage)

Blameless. Every cause below is a *system* property; every action item is a system
change with an owner, not "be more careful".

## What happened

Building the rigorous eval (bootstrap CIs + ECE + error analysis) surfaced that the
multi-model consensus is **mis-calibrated toward the middle**: it never commits a
forecast above ~0.585, discriminates weakly between YES- and NO-resolving questions,
and beats the trivial base-rate baseline by an amount that is **not statistically
distinguishable at n=50**. The engine "ran and produced numbers" the whole time —
what was missing was the measurement that made the weakness visible.

## Impact (in numbers)

- Consensus **Brier 0.223**, 95% CI **[0.183, 0.264]** — the CI **overlaps** the
  base-rate Brier **0.230**, i.e. skill vs base-rate **+0.032** is not significant.
- **ECE 0.106** — meaningful miscalibration.
- Forecast range **0.033–0.585** — never confident on YES.
- Mean forecast **0.416 on YES-resolving** vs **0.355 on NO-resolving** questions —
  ~6 points of separation where a good forecaster would show far more.
- Dominant error buckets: `hedged_wrong` (12) + `missed_yes` (7) — wrong-side calls
  from the mushy middle. (`eval/results/error_analysis.md`.)

## Timeline

1. Eval harness existed (Brier + base-rate + 5-bin calibration) but reported only a
   point estimate — the +0.007 gap over base-rate looked like a (weak) win.
2. Added `bootstrap_ci`/`ECE`/`brier_skill_score` → the CI overlap made "beats
   base-rate" indefensible.
3. `error_analysis.py` open-coded the worst calls → the ≤0.585 ceiling and weak
   YES/NO separation named the failure as **under-confidence / regression to the mean**.

## Root causes (systemic, multiple)

1. **Prompt pulls toward the market.** The estimation prompt says "most of the time
   your estimate should be within 15% of the market price" and clamps to [0.02, 0.98]
   — both compress the output distribution.
2. **Ensemble averaging regresses to the mean.** A weighted average of several models
   is structurally less extreme than its inputs; the `extremizing_factor` (1.15) is
   too weak to counteract it.
3. **Calibration was never a gate.** Nothing failed when the forecaster hedged; the
   only metric watched was a point-estimate Brier, which hid it.
4. **n=50 is too small** to detect a real edge — wide CIs are a *measurement* limit,
   not only a model flaw. We were reading noise as signal.

## Action items

| # | Change (system, not effort) | Owner | Status |
|---|---|---|---|
| 1 | Add a calibration/extremizing post-processing step, tuned on a **dev split**, report on holdout | eval | open |
| 2 | Extend the CI regression gate to also bound **ECE** (not just Brier) so calibration can't silently regress | ci | open |
| 3 | Get a **single-model baseline + larger n** via the keyed run for a CI-supported comparison | eval | pending keyed run |
| 4 | Re-examine the "stay within 15% of market" prompt instruction (it may be the biggest compressor) — as an ADR | design | open |

## Lessons

- **Measure calibration explicitly (ECE + reliability curve), not just Brier** — a
  Brier that looks "fine" hid a systematic hedge.
- **At small n, report the CI and believe it.** A point estimate that beats a baseline
  by a hair, with an overlapping CI, is not a win — saying so is the honest result.
