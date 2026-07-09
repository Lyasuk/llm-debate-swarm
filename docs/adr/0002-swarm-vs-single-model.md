# ADR-0002 — N-agent debate swarm vs single-model self-consistency

**Status:** Accepted (provisional) · **Date:** 2026-07-09

## Context

The headline hypothesis is that a multi-agent debate produces a *better-calibrated*
probability than a single model. But a full combined forecast is ~210 LLM calls
(30 agents × 7 rounds) + 3 consensus calls vs **1** for a single-model baseline —
two orders of magnitude more cost/latency. That cost has to be earned.

## Decision

Keep the swarm, but treat the "swarm beats single-model" claim as a **falsifiable
hypothesis with a kill-number**, not a given. The eval harness scores the swarm
against a single-model@temp0 baseline and the always-base-rate baseline on a
holdout; the debate is justified only if it beats them by enough Brier that the
bootstrap CI supports it.

## Consequences

- **Pay:** ~200× the calls/latency of a single call; the dominant cost driver.
- **Get:** a debiasing structure (blind rounds, devil's advocate, pre-mortem,
  robust aggregation) that is defensible *as a reliability/eval showcase* even
  before it wins on cost.
- **Honest status:** at n=50 the *consensus* arm only marginally beats base-rate
  (skill +0.032, CI overlaps). The 30-agent swarm arm's calibration is **not yet
  measured on a paid run** — labelled pending in the eval, not faked.

## Revisit when

- The eval (larger n, paid swarm run) shows the swarm does **not** beat the
  single-model baseline by a CI-supported margin → drop the swarm, ship the
  single call, keep the eval.
- Volume/latency requirements make ~200 calls/forecast infeasible → same.
