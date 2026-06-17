"""Compiles raw research data into a structured document for LLM/MiroFish."""

from __future__ import annotations

from datetime import datetime

from llm_debate_swarm.config import AppConfig
from llm_debate_swarm.research.research_manager import ResearchContext
from llm_debate_swarm.utils.logger import get_logger

log = get_logger("research.compiler")


class DocumentCompiler:
    """Assembles a structured research document from raw research data."""

    def __init__(self, config: AppConfig):
        self.max_chars = config.research.document_max_chars

    def compile(self, context: ResearchContext) -> str:
        """Compile research context into a structured markdown document."""
        market = context.market
        sections: list[str] = []

        # Header
        sections.append(f"# Research Brief: {market.question}\n")

        # Current market state
        sections.append(self._section_market_state(market))

        # Weather forecast data (if applicable) — HIGH edge source, put near top
        if context.weather_section:
            sections.append(context.weather_section)

        # Web search findings
        if context.web_search:
            sections.append(self._section_web_findings(context))

        # Contrarian arguments (always include)
        sections.append(self._section_contrarian(market))

        # Key entities
        sections.append(self._section_entities(context))

        # Assemble and truncate
        doc = "\n".join(sections)
        if len(doc) > self.max_chars:
            doc = doc[: self.max_chars - 50] + "\n\n[Document truncated for length]"

        log.info(f"Compiled document: {len(doc)} chars for '{market.question[:50]}...'")
        return doc

    def _section_market_state(self, market) -> str:
        lines = [
            "## Current Market State",
            f"- **Polymarket YES price**: {market.yes_price:.1%}",
            f"- **Polymarket NO price**: {market.no_price:.1%}",
            f"- **Volume 24h**: ${market.volume_24h:,.0f}",
            f"- **Total volume**: ${market.volume:,.0f}",
            f"- **Liquidity**: ${market.liquidity:,.0f}",
            f"- **Spread**: {market.spread:.3f}",
            f"- **Days to resolution**: {market.days_to_resolution:.0f}",
            f"- **Category**: {market.category}",
            f"- **Resolution source**: {market.resolution_source}",
            f"- **Research date**: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
            "",
        ]
        return "\n".join(lines)

    def _section_web_findings(self, context: ResearchContext) -> str:
        lines = ["## Key Facts & Evidence\n"]

        for search_output in context.web_search:
            # Include Tavily's AI answer if available
            if search_output.answer:
                lines.append(f"**Summary**: {search_output.answer}\n")

            # Include top results
            for result in search_output.results[:5]:
                if result.content:
                    # Truncate each result to ~500 chars
                    content = result.content[:500]
                    lines.append(f"- **{result.title}** ({result.url})")
                    lines.append(f"  {content}")
                    lines.append("")

        return "\n".join(lines)

    def _section_contrarian(self, market) -> str:
        """Prompt section that forces balanced analysis."""
        price = market.yes_price
        if price > 0.5:
            direction = "NO (against the majority)"
            opposite = f"{1 - price:.1%}"
        else:
            direction = "YES (against the majority)"
            opposite = f"above {price:.1%}"

        return (
            "## Contrarian Considerations\n"
            f"The market currently prices this at {price:.1%} YES. "
            f"Consider arguments for {direction}:\n"
            f"- What would need to happen for the probability to move to {opposite}?\n"
            f"- What information might the market be overlooking?\n"
            f"- What historical precedents suggest a different outcome?\n"
        )

    def _section_entities(self, context: ResearchContext) -> str:
        """Extract key entities from research for MiroFish GraphRAG."""
        entities: set[str] = set()

        for search_output in context.web_search:
            for result in search_output.results:
                # Simple entity extraction: capitalized multi-word phrases
                words = result.title.split()
                for i, word in enumerate(words):
                    if word and word[0].isupper() and len(word) > 2:
                        # Try to get multi-word entity
                        entity_parts = [word]
                        for j in range(i + 1, min(i + 3, len(words))):
                            if words[j] and words[j][0].isupper():
                                entity_parts.append(words[j])
                            else:
                                break
                        if len(entity_parts) >= 2:
                            entities.add(" ".join(entity_parts))

        if not entities:
            return ""

        lines = ["## Key Entities\n"]
        for entity in sorted(entities)[:20]:
            lines.append(f"- {entity}")

        return "\n".join(lines)
