"""Command-line interface for llm-debate-swarm."""
from __future__ import annotations

import asyncio

import click
from opentelemetry import trace
from rich.console import Console
from rich.table import Table

from llm_debate_swarm import __version__
from llm_debate_swarm.engine import DebateSwarmEngine
from llm_debate_swarm.graph import forecast_with_graph
from llm_debate_swarm.obs.tracing import setup_tracing

console = Console()


@click.group()
def cli() -> None:
    """llm-debate-swarm — multi-LLM debate & arbitration engine."""


@cli.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"llm-debate-swarm {__version__}")


@cli.command()
@click.argument("question")
@click.option("--category", default="", help="Optional category hint.")
@click.option("--prior", type=float, default=None, help="Market/expert prior P(YES) in [0,1].")
@click.option("--horizon", "horizon_days", type=float, default=None, help="Days until resolution.")
@click.option("--no-consensus", is_flag=True, help="Disable the multi-LLM consensus stage.")
@click.option("--no-swarm", is_flag=True, help="Disable the debate-swarm stage.")
@click.option("--research", is_flag=True, help="Enable Tavily web research (needs TAVILY_API_KEY).")
@click.option("--swarm-weight", type=float, default=0.5, show_default=True,
              help="Blend weight of swarm vs consensus (0=consensus only, 1=swarm only).")
@click.option("--engine", "engine_kind", type=click.Choice(["graph", "async"]),
              default="graph", show_default=True,
              help="Orchestrator: 'graph' = LangGraph (primary), 'async' = raw asyncio.")
def forecast(
    question: str,
    category: str,
    prior: float | None,
    horizon_days: float | None,
    no_consensus: bool,
    no_swarm: bool,
    research: bool,
    swarm_weight: float,
    engine_kind: str,
) -> None:
    """Forecast a binary QUESTION and print a calibrated verdict."""
    setup_tracing()  # no-op unless LANGFUSE_*/LANGSMITH_*/OTEL_* env vars are set
    # Web research lives on the async engine only; fall back to it when asked.
    if research and engine_kind == "graph":
        console.print("[yellow]--research uses the async engine; switching --engine async.[/]")
        engine_kind = "async"

    if engine_kind == "graph":
        verdict = asyncio.run(forecast_with_graph(
            question, category=category, prior=prior, horizon_days=horizon_days,
            use_consensus=not no_consensus, use_swarm=not no_swarm, swarm_weight=swarm_weight,
        ))
    else:
        engine = DebateSwarmEngine(
            use_consensus=not no_consensus,
            use_swarm=not no_swarm,
            research=research,
            swarm_weight=swarm_weight,
        )
        verdict = asyncio.run(
            engine.forecast(question, category=category, prior=prior, horizon_days=horizon_days)
        )

    console.print()
    console.print(f"[bold]Q:[/] {verdict.question}")
    console.print(
        f"[bold green]P(YES) = {verdict.probability:.1%}[/]  "
        f"confidence={verdict.confidence:.0%}  type={verdict.question_type}"
    )
    if verdict.consensus_probability is not None and verdict.swarm_probability is not None:
        console.print(
            f"  consensus={verdict.consensus_probability:.1%}  "
            f"swarm={verdict.swarm_probability:.1%}  "
            f"disagreement={verdict.disagreement:.1%}"
        )
    if verdict.agent_count:
        console.print(
            f"  swarm: {verdict.agent_count} agents / {verdict.rounds_completed} rounds, "
            f"anchoring_shift={verdict.anchoring_shift:+.1%}, "
            f"convergence={verdict.convergence_ratio:.0%}, cost=${verdict.cost_usd:.4f}"
        )
    if verdict.per_model:
        table = Table("model", "provider", "P(YES)", "error")
        for m in verdict.per_model:
            table.add_row(
                str(m["model"]), str(m["provider"]),
                f"{m['probability']:.1%}", (m["error"] or "")[:40],
            )
        console.print(table)

    # BatchSpanProcessor exports in a background thread; flush before the
    # process exits or OTLP backends silently lose the run's spans.
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()


if __name__ == "__main__":
    cli()
