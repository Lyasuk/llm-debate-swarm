# ADR-0004 — Statistical meta-synthesis vs an LLM meta call

**Status:** Accepted · **Date:** 2026-07-09

## Context

After the swarm's 7 rounds, the per-agent estimates must be fused into one number.
An LLM "meta-synthesis" call was tried; it **anchored on outliers** (observed: a
0.99 extreme dragged the meta to 0.125 while the trimmed mean was 0.32).

## Decision

Fuse with **robust statistics** — trimmed mean + MAD outlier rejection + bimodal
detection — not an LLM call. Deterministic, outlier-resistant, and **zero extra
API cost**.

## Consequences

- **Pay:** no free-text "reasoning" for the final fusion (we keep per-agent
  reasoning instead); statistics can't capture a genuinely novel argument one
  agent raises.
- **Get:** stability, reproducibility, no anchoring, and one fewer paid call on the
  hot path.

## Revisit when

- Eval shows the statistical fusion is systematically mis-calibrated in a way a
  guided LLM meta (with the outlier problem fixed) would beat — measure before switching.
