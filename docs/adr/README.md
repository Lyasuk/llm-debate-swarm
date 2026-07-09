# Architecture Decision Records

Short [Michael-Nygard-format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
records of the load-bearing decisions. Each carries a **Revisit when** trigger —
the condition under which the decision should be re-opened. ADRs are immutable:
to change a decision, add a new ADR that supersedes the old one.

| # | Decision | Status |
|---|---|---|
| [0001](0001-langgraph-as-primary-orchestrator.md) | LangGraph as the primary orchestrator | Accepted |
| [0002](0002-swarm-vs-single-model.md) | N-agent debate swarm vs single-model self-consistency | Accepted (provisional) |
| [0003](0003-vendor-neutral-dual-tracing.md) | Vendor-neutral tracing → Langfuse + LangSmith | Accepted |
| [0004](0004-statistical-meta-vs-llm-meta.md) | Statistical meta-synthesis vs an LLM meta call | Accepted |
| [0005](0005-market-price-baseline-limitation.md) | Market-price baseline is deferred, not faked | Accepted |
