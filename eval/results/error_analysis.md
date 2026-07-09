# Error analysis — `consensus` (n=50)

Diagnosis of *where* the forecaster is wrong, from committed predictions (no API calls). Fix the dominant category, then re-measure the delta.

## Prediction range
- forecasts span **0.033 – 0.585**
- mean forecast on questions that resolved **YES**: **0.416**
- mean forecast on questions that resolved **NO**: **0.355**

## Failure buckets
- **7** — missed YES (resolved YES, forecast < 0.40 — under-forecast the event)
- **0** — missed NO (resolved NO, forecast > 0.60 — over-forecast the event)
- **12** — hedged wrong (0.40–0.60, landed on the wrong side of 50%)
- **31** — correct side of 50%

**Dominant failure mode:** `hedged_wrong` — hedged wrong (0.40–0.60, landed on the wrong side of 50%)

## Worst calls (highest squared error)

| squared err | forecast | resolved | question |
|---:|---:|---:|---|
| 0.621 | 0.212 | 1 | US forces enter Iran by April 30? |
| 0.562 | 0.250 | 1 | US strikes Iran by February 28, 2026? |
| 0.561 | 0.251 | 1 | Khamenei out as Supreme Leader of Iran by March 31? |
| 0.531 | 0.271 | 1 | Khamenei out as Supreme Leader of Iran by February 28? |
| 0.433 | 0.342 | 1 | Russia x Ukraine ceasefire by May 31, 2026? |
| 0.416 | 0.355 | 1 | U.S. anti-cartel ground operation in Mexico by January 31? |
| 0.385 | 0.380 | 1 | US x Iran ceasefire by April 7? |
| 0.339 | 0.583 | 0 | Strait of Hormuz traffic returns to normal by end of April? |
