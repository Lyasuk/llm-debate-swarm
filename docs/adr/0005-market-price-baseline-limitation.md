# ADR-0005 — Market-price baseline is deferred, not faked

**Status:** Accepted · **Date:** 2026-07-09

## Context

The market-implied price is the strong baseline a forecaster should be measured
against. The eval dataset is *resolved* Polymarket markets, keyed by `market_id`.

## Decision

**Do not** compute a market-price baseline from resolved markets' **final** prices.
A resolved market's final price ≈ its outcome (0/1), so scoring it would be a
trivially near-perfect, **contaminated** baseline that makes the forecaster look
worse for no honest reason. A fair market baseline needs the market's
**point-in-time** price at forecast time (Polymarket price history), which we have
not wired. `eval/analyze.py` therefore reports the market-price baseline as
**pending** and accepts an optional `--market-prices` file when a valid
point-in-time set exists.

## Consequences

- **Pay:** the eval currently lacks its strongest baseline.
- **Get:** we don't ship a misleading number. Recognizing that a resolved final
  price is contaminated is itself the point.

## Revisit when

- We wire Polymarket price-history to snapshot each market's price at a fixed lead
  time before resolution → add it as a real baseline (and choose/justify the lead time).
