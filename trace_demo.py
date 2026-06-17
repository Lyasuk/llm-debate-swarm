"""Run one forecast with OpenTelemetry tracing on, then dump + render the trace.

Writes docs/sample_trace.json (raw spans) and docs/sample_trace.svg (waterfall).
Set OTEL_EXPORTER_OTLP_ENDPOINT to ALSO stream the same spans to Langfuse / Jaeger
/ Grafana Tempo (pip install 'llm-debate-swarm[otlp]').

Usage:
    python trace_demo.py "Will X happen by Y?"      # consensus only (cheap)
    python trace_demo.py "..." --swarm              # also run the debate swarm
"""
from __future__ import annotations

import asyncio
import os
import sys

from llm_debate_swarm.engine import DebateSwarmEngine
from llm_debate_swarm.obs.render import render_file
from llm_debate_swarm.obs.tracing import dump_spans, setup_tracing


async def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    question = args[0] if args else "Will global average CO2 exceed 430 ppm before 2030?"
    use_swarm = "--swarm" in sys.argv

    setup_tracing()
    engine = DebateSwarmEngine(use_consensus=True, use_swarm=use_swarm)
    v = await engine.forecast(question)
    print(f"P(YES)={v.probability:.2f}  confidence={v.confidence:.0%}  "
          f"consensus={v.consensus_probability}  swarm={v.swarm_probability}")

    os.makedirs("docs", exist_ok=True)
    n = dump_spans("docs/sample_trace.json")
    render_file("docs/sample_trace.json", "docs/sample_trace.svg")
    print(f"captured {n} spans -> docs/sample_trace.json + docs/sample_trace.svg")


if __name__ == "__main__":
    asyncio.run(main())
