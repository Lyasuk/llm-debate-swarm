# ADR-0001 — LangGraph as the primary orchestrator

**Status:** Accepted · **Date:** 2026-07-09

## Context

The forecast fans out to N model calls + a swarm, then fuses them. The original
code did this with a raw `asyncio.gather` in the engine, and carried a *separate*,
never-invoked LangGraph "surface" for show. Two orchestrations, one of them dead.
We need one real orchestration that also gives durable execution, per-node
observability, and explicit control flow a reviewer can read.

## Options considered

- **Raw `asyncio.gather` only** — simplest, but no checkpointing, no standard
  trace of the execution path, and the graph is decorative.
- **Role-based framework (CrewAI)** — fast to demo; weak checkpointing, role-play
  token overhead, less control over aggregation.
- **LangGraph `StateGraph` as the real path** — explicit state machine, Send
  fan-out, reducers, deferred fan-in, pluggable checkpointer.

## Decision

Make the LangGraph `StateGraph` the **primary** path (the CLI defaults to it) and
keep the async engine as the low-level primitive its nodes reuse (`_query_model`,
`_build_consensus`, `combine_verdict`) — one implementation, two surfaces. Model
the consensus panel as a **Send** fan-out with a reducer on the predictions
channel; fuse with a deferred `combine` node. Target `langgraph>=1.2,<2`
(1.0 GA changed the API vs 0.x); do **not** use the deprecated `create_react_agent`
— we need custom deterministic control flow, not a prebuilt agent loop.

## Consequences

- **Pay:** a heavier dependency and LangGraph's learning curve; the swarm's 7-round
  debate is still opaque inside one node (not lifted into graph nodes — deliberate,
  to avoid rewriting 1.3k LOC of working debiasing logic).
- **Get:** durable execution (proven by a kill/resume test), a standard trace of
  which branch ran, budgets (`recursion_limit` + per-run timeout), and an honest
  "who owns control flow" story.

## Revisit when

- We need the individual debate rounds to be independently resumable/observable
  (then lift them into graph nodes), or
- LangGraph 2.0 lands (re-check the API surface), or
- the dependency weight stops being worth it for a single fan-out/fan-in shape.
