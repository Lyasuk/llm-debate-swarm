# llm-debate-swarm

**A multi-LLM debate & arbitration engine.** Give it a binary question; it runs a
weighted **multi-provider LLM consensus** *and* an **N-agent, multi-round debiasing
debate swarm** in parallel, then aggregates them with robust statistics into a single
**calibrated probability** with confidence and an explicit cross-method **disagreement**
signal.

Built as async Python, provider-agnostic (Anthropic · OpenAI · Google · Groq), with
per-call cost tracking and an optional SQLite audit of every model and agent.

> This repo is the domain-neutral core extracted from a private probabilistic-forecasting
> engine. The trading/exchange layer that consumed these probabilities is intentionally
> **not** included — this is the reasoning engine, not a trading bot.

---

## Why it exists

Single-model self-evaluation shares the model's own blind spots. Asking three models and
averaging barely helps — they anchor on each other and on whatever number they see first.
This engine attacks that directly:

- **Independence first.** Agents produce *blind* estimates before seeing any anchor
  (market price, peers' answers, or the question's framing).
- **Structured debiasing.** A 7-round protocol: blind → debate (blind, then anchor-aware)
  → **pre-mortem** ("assume your estimate was wrong — what did you miss?") → meta-synthesis.
- **Adversarial pressure.** A configurable set of agents argue the *devil's-advocate* side.
- **Groupthink detection.** If the swarm collapses to too few distinct estimates, the round
  is re-run at higher temperature to restore diversity.
- **Robust aggregation.** Trimmed mean + MAD outlier rejection, with anchoring-shift and
  convergence metrics surfaced so you can see *how* the answer was reached, not just the number.

The multi-provider consensus and the swarm are computed independently; their **disagreement**
is a first-class output — high disagreement is a signal, not noise.

## Architecture

```
question
  ├─ classify (rule-based question typing: barrier / deadline / fixed-date / head-to-head)
  ├─ (optional) web research  ──────────────► research document
  │
  ├─ PARALLEL ─┬─ multi-LLM weighted consensus   (N providers, weighted, structured JSON)
  │            └─ debate swarm                    (M agents × 7 debiasing rounds)
  │                 ├─ blind rounds (anti-anchoring)
  │                 ├─ debate rounds (blind, then anchor-aware)
  │                 ├─ devil's-advocate agents
  │                 ├─ pre-mortem round
  │                 ├─ groupthink detection + retry
  │                 └─ robust aggregation (trimmed mean, MAD, anchoring/convergence)
  │
  └─ combine ─► Verdict { probability, confidence, consensus_p, swarm_p,
                          disagreement, anchoring_shift, convergence, cost, per_model }
```

## Quickstart

```bash
git clone https://github.com/Lyasuk/llm-debate-swarm
cd llm-debate-swarm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # add at least one provider API key

debate-swarm forecast "Will global average CO2 exceed 430 ppm before 2030?"
debate-swarm forecast "Lakers vs Celtics — will the Lakers win?" --no-swarm
```

Programmatic use:

```python
import asyncio
from llm_debate_swarm import DebateSwarmEngine

async def main():
    engine = DebateSwarmEngine(use_swarm=True)
    v = await engine.forecast("Will SpaceX reach orbit with Starship by Q4?", horizon_days=120)
    print(f"{v.probability:.1%}  (consensus={v.consensus_probability}, "
          f"swarm={v.swarm_probability}, disagreement={v.disagreement:.1%})")

asyncio.run(main())
```

Every stage is optional and degrades gracefully: disable either stage, or run with whatever
provider keys you have — missing providers are skipped, not fatal.

## Configuration

`config.yaml` controls the consensus model panel (names, providers, weights) and the swarm
(agent count, rounds, devil's-advocate count, trimming, bucketed model routing). See the
file for inline docs.

## Tests

```bash
pytest            # unit tests (types, classifier, import graph) — no network/keys required
```

## Roadmap

- [ ] **Evaluation harness** — Brier score + calibration curve over a public set of resolved
      questions (the engine already computes the calibration internals).
- [ ] **Observability** — OpenTelemetry spans + Langfuse traces of every round and model call.
- [ ] **LangGraph** orchestration wrapper for the swarm graph.
- [ ] FastAPI service + Dockerfile.

## License

MIT — see [LICENSE](LICENSE).
