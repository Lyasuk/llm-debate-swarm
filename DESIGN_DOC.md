# Design doc — llm-debate-swarm

**Status:** approved · **Date:** 2026-07-09 · **Author:** Vladyslav Liaskovych

> Written before-the-build framing. The point of this doc is to make the design
> decisions *disagreeable* — each one is stated plainly enough that a reviewer
> can point at it and argue. Decision records live in [`docs/adr/`](docs/adr/).

## 1. Problem

Given a **binary market/event question**, produce a **calibrated probability**
of YES that a decision-maker can act on, with the reasoning attached. A single
LLM call is fast but blind-spotted: it anchors, it herds toward the obvious
answer, and — critically for a forecast — it is usually **mis-calibrated** (its
70% is not right 70% of the time).

## 2. Goal & non-goals

**Goal.** A calibrated binary-probability forecaster whose quality is *measured*
(not asserted), whose runs are *observable* per LLM call, and whose orchestration
is explicit and resumable.

**Non-goals (explicit).**
- **Not** a general chatbot or open-ended agent — control flow is fixed code (see §4).
- **Not** trade execution or position sizing — this is an AI-engineering /
  reliability showcase, not a trading strategy. It emits a probability; it never
  moves money.
- **Not** calibration on non-binary or continuous outcomes.
- **Not** a large-scale service — see the cut-list in [`PHASE2_HARDENING_SPEC.md`](PHASE2_HARDENING_SPEC.md).

## 3. Proposed solution

Two independent estimators, fused deterministically:

1. **Multi-LLM weighted consensus** — N provider models answer the same question
   in parallel; a weighted aggregate + a spread-based confidence. (The exercised,
   evaluated path.)
2. **N-agent debiasing debate swarm** — 30 agents over a 7-round protocol
   (blind → debate → pre-mortem → final), devil's-advocate injection, groupthink
   detection, trimmed-mean/MAD statistical meta-synthesis.

```
question ─▶ classify ─▶ ┌ consensus (N models, Send fan-out) ┐ ─▶ combine ─▶ P(YES)
                        └ swarm (30 agents × 7 rounds)        ┘
```

Orchestrated as an explicit LangGraph `StateGraph` ([ADR-0001](docs/adr/0001-langgraph-as-primary-orchestrator.md)).

## 4. Who decides control flow (why this is a workflow, not an "agent")

The **graph** decides what runs next — fan-out count, aggregation, fusion are all
deterministic code. The LLMs live *only inside* the `debater`/`swarm` node bodies.
There is no point where a model freely chooses the next step. So by the standard
test ("who decides what happens next — code or the model?") this is a
**workflow / orchestrated multi-agent system**, and the README says so. The word
"agent" in the docs is reserved for the debater personas, which are node content.

## 5. Alternatives considered

| Option | Why not (for this problem) |
|---|---|
| **No AI** (use the market's own price) | The strong baseline, and honestly hard to beat — but it only exists for questions that are already priced; the tool must also handle un-priced events. Kept as a baseline, not the product. See [ADR-0005](docs/adr/0005-market-price-baseline-limitation.md). |
| **Single strong-model call @ temp 0** | The right default for most tasks and the baseline to beat. We keep it as the eval baseline. The bet of this project is that debate/ensemble improves *calibration*; that bet is **not yet proven at n=50** (§6). |
| **Role-based framework (CrewAI)** | Faster to a demo, but weak checkpointing and role-play token overhead; we need deterministic aggregation, durable execution, and per-call cost — graph-based fits. [ADR-0001](docs/adr/0001-langgraph-as-primary-orchestrator.md). |
| **LLM meta-synthesis of the swarm** | Anchors on outliers; replaced by robust statistics (trimmed mean + MAD). [ADR-0004](docs/adr/0004-statistical-meta-vs-llm-meta.md). |

## 6. Quality bar (an SLO, fixed before tuning)

Quality is a **proper scoring rule on a holdout of resolved binary questions**,
not "looks reasonable":

- **Primary:** Brier score, reported with a bootstrap **95% CI** and n.
- **Calibration:** reliability curve + **ECE**.
- **Skill:** Brier skill vs the always-base-rate baseline, and (when available)
  vs a single-model baseline.
- **Error asymmetry:** a confident-and-wrong forecast costs more than a hedged one.
- **Contract:** we do not claim "beats baseline" unless the CI supports it.

**Current measured result (n=50 resolved Polymarket questions, consensus):**
Brier **0.223** (95% CI **[0.183, 0.264]**), ECE **0.106**, skill vs base-rate
**+0.032**. **Honest reading: the CI overlaps the base-rate Brier (0.230), so at
n=50 the improvement is not statistically distinguishable.** Reproduce with
`python -m eval.analyze`. Error analysis (`eval/results/error_analysis.md`) shows
the forecaster **compresses toward the middle** (never exceeds ~0.585) and
discriminates weakly (mean forecast 0.416 on YES vs 0.355 on NO) — an
under-confidence / regression-to-the-mean failure, fixable by stronger
extremizing rather than swapping models.

## 7. When a debate swarm is the WRONG tool

A full combined forecast is **~210 LLM calls** (30 agents × 7 rounds) + 3 consensus
calls, versus **1** for a single-model baseline. That is ~2 orders of magnitude
more cost and latency. The falsifiable bet: debate must beat the single-model
baseline by enough Brier to justify that. **We have not shown it does at n=50.**
Therefore: for high-volume or latency-sensitive use, **use the single-model or
market-price baseline**; reserve the swarm for low-volume, decision-heavy,
offline questions where calibration matters more than cost. Building the swarm
was justified as a *reliability/eval showcase*; deploying it at scale is not, yet.

## 8. Pre-mortem (how this fails)

1. **Groupthink** — debaters converge, so debate adds no signal over one call.
   *Mitigation:* diversity check + re-run at higher temp; devil's advocates.
2. **Aggregator overconfidence** — meta systematically over/under-shoots.
   *Mitigation:* robust trimmed-mean; measured by ECE; **currently under-confident**.
3. **Provider stall** — one slow/hung model stalls the whole graph.
   *Mitigation:* per-run timeout budget + graceful degradation (a failed arm
   drops out; the other still produces a verdict).
4. **Calibration drift** — well-calibrated in-sample, drifts on new categories.
   *Mitigation:* pinned model versions + the CI regression gate; accepted risk at n=50.
5. **Eval leakage** — some questions predate model cutoffs. *Mitigation:* the
   consensus-vs-baseline comparison is leakage-robust (all configs see the same
   questions); documented in `eval/questions.yaml`.

## 9. Cost & scale (back-of-envelope)

- Consensus: 3 calls/forecast (Haiku + 2 Groq). Groq free-tier; Anthropic Haiku ~$1/$5 per 1M tok.
- Swarm: ~210 cheap-model calls/forecast; the dominant cost/latency driver.
- **Prompt caching** applies: every debater shares a large static prefix
  (role + rubric), variable question at the tail — "static forward, variable back".
- **Economic breaking point:** above roughly a few hundred forecasts/day the swarm's
  cost/latency stops being viable versus a single strong model; that is the line at
  which this design should be replaced, not scaled.

## 10. Observability & security

Every LLM call is a span with model + tokens + cost + latency (OTel GenAI
conventions), fanned out to Langfuse (self-host) **and** LangSmith from one
instrumentation ([tracing](src/llm_debate_swarm/obs/tracing.py)). Threat model
(indirect prompt injection via untrusted market/evidence text; output validation)
is in [`SECURITY.md`](SECURITY.md).
