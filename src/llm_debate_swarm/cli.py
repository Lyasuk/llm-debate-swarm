"""Command-line interface for llm-debate-swarm."""
from __future__ import annotations

import asyncio

import click
from rich.console import Console
from rich.table import Table

from llm_debate_swarm import __version__
from llm_debate_swarm.engine import DebateSwarmEngine

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
def forecast(
    question: str,
    category: str,
    prior: float | None,
    horizon_days: float | None,
    no_consensus: bool,
    no_swarm: bool,
    research: bool,
    swarm_weight: float,
) -> None:
    """Forecast a binary QUESTION and print a calibrated verdict."""
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


if __name__ == "__main__":
    cli()
