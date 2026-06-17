"""Orchestrates all research sources for a given market question."""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_debate_swarm.analysis.question_classifier import QuestionClassification, QuestionType
from llm_debate_swarm.config import AppConfig
from llm_debate_swarm.research.sources.tavily_search import TavilySearch, TavilySearchOutput
from llm_debate_swarm.types import Question as Market
from llm_debate_swarm.utils.logger import get_logger

log = get_logger("research.manager")


@dataclass
class ResearchContext:
    """All research data gathered for a single market."""

    market: Market
    web_search: list[TavilySearchOutput] = field(default_factory=list)
    classification: QuestionClassification | None = None
    weather_section: str = ""  # Pre-formatted weather data (Open-Meteo)
    # Phase 3 additions:
    # news_articles: list[NewsArticle] = field(default_factory=list)
    # reddit_sentiment: SentimentData | None = None
    # metaculus_forecast: float | None = None
    # academic_papers: list[Paper] = field(default_factory=list)


class ResearchManager:
    """Gathers comprehensive research for a market question."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._tavily: TavilySearch | None = None

    def _get_tavily(self) -> TavilySearch:
        if self._tavily is None:
            src_cfg = self.config.research.sources.get("tavily")
            max_results = src_cfg.max_results if src_cfg else 10
            self._tavily = TavilySearch(max_results=max_results)
        return self._tavily

    async def research_market(
        self,
        market: Market,
        classification: QuestionClassification | None = None,
    ) -> ResearchContext:
        """Run deep research on a market with type-aware query generation."""
        log.info(f"Researching: {market.question[:80]}...")
        context = ResearchContext(market=market, classification=classification)

        # Specialized data sources based on question content
        # Weather markets: fetch Open-Meteo forecast (HIGH edge source)
        if self._is_weather_question(market.question):
            try:
                from llm_debate_swarm.research.sources.weather_api import (
                    format_weather_for_research,
                    get_weather_context,
                )
                weather_ctx = await get_weather_context(market.question)
                if weather_ctx:
                    context.weather_section = format_weather_for_research(
                        weather_ctx, market.question
                    )
                    log.info(
                        f"Weather data: P50={weather_ctx.forecast.temp_max_p50:.1f}°C "
                        f"from {weather_ctx.forecast.models_count} models"
                    )
            except Exception as e:
                log.warning(f"Weather API failed: {e}")

        # Phase 1: Web search via Tavily
        tavily_cfg = self.config.research.sources.get("tavily")
        if tavily_cfg and tavily_cfg.enabled:
            context.web_search = await self._research_web(market, classification)

        log.info(
            f"Research complete: {len(context.web_search)} web searches"
            + (" + weather" if context.weather_section else "")
        )
        return context

    def _is_weather_question(self, question: str) -> bool:
        """Quick check if question is about weather/temperature."""
        q = question.lower()
        return any(w in q for w in [
            "temperature", "weather", "hurricane", "typhoon", "snow",
            "rainfall", "precipitation", "heat wave", "°c", "°f",
        ])

    async def _research_web(
        self,
        market: Market,
        classification: QuestionClassification | None,
    ) -> list[TavilySearchOutput]:
        """Generate smart type-aware search queries and run them."""
        tavily = self._get_tavily()
        queries = self._generate_queries(market, classification)
        results = await tavily.multi_search(queries)

        # Логуємо кожний search у research_artifacts table
        decision_id = getattr(market, "_decision_id", None)
        if decision_id is not None:
            try:
                from llm_debate_swarm.tracking.decision_logger import get_logger_instance
                from llm_debate_swarm.utils.cost_tracker import get_tracker
                dl = get_logger_instance()
                ct = get_tracker()

                # Classify query type for each query
                query_types = []
                for i, q in enumerate(queries):
                    if "opposing" in q.lower() or "unlikely" in q.lower():
                        query_types.append("opposing")
                    elif "polls" in q.lower() or "betting" in q.lower():
                        query_types.append("calibration")
                    else:
                        query_types.append("primary")

                for q, qt, r in zip(queries, query_types, results):
                    top_urls = []
                    ai_answer = ""
                    results_count = 0
                    error_msg = ""
                    raw_len = 0
                    if r is not None:
                        try:
                            top_urls = [
                                {"url": s.url[:200], "title": s.title[:200],
                                 "snippet": s.content[:300]}
                                for s in (r.results or [])[:5]
                            ]
                            results_count = len(r.results) if r.results else 0
                            ai_answer = (r.answer or "")[:2000]
                            raw_len = sum(len(s.content or "") for s in (r.results or []))
                        except Exception:
                            pass

                    dl.log_research(
                        decision_id=decision_id,
                        search_query=q,
                        query_type=qt,
                        results_count=results_count,
                        top_urls=top_urls,
                        ai_answer=ai_answer,
                        raw_content_len=raw_len,
                        cost_usd=0.008,  # Tavily flat fee per search
                        error=error_msg,
                        cached=False,
                    )
                    ct.record_flat(
                        provider="tavily",
                        role="research",
                        cost_usd=0.008,
                    )
            except Exception as log_exc:
                log.warning(f"Research logging failed: {log_exc}")

        return results

    def _generate_queries(
        self,
        market: Market,
        classification: QuestionClassification | None,
    ) -> list[str]:
        """Generate 3 diverse type-aware search queries.

        Different question types need different kinds of research:
        - BARRIER: price history, technical levels, volatility
        - FIXED_DATE_EVENT: polls, betting markets, recent news
        - DEADLINE_EVENT: news, statements, institutional positions
        - HEAD_TO_HEAD: recent form, injuries, head-to-head record
        """
        question = market.question

        # Fallback: no classification, use generic queries
        if classification is None or classification.question_type == QuestionType.UNKNOWN:
            return self._generic_queries(market)

        qt = classification.question_type

        if qt == QuestionType.BARRIER:
            asset = classification.asset or "the asset"
            level = classification.level
            direction = classification.direction or "reach"
            level_str = f"${level:,.0f}" if level else "target level"
            queries = [
                # 1. Current price + recent trend
                f"{asset} price today current value 2026",
                # 2. Historical volatility and analyst forecasts
                f"{asset} price forecast analysis volatility {level_str} April 2026",
                # 3. Technical analysis + support/resistance
                f"{asset} technical analysis support resistance {level_str}",
            ]

        elif qt == QuestionType.FIXED_DATE_EVENT:
            queries = [
                # 1. Direct question + recent polls
                f"{question} latest polls 2026",
                # 2. Betting markets and forecasts
                f"{question} betting odds forecast prediction",
                # 3. OPPOSING VIEW — balanced input (Fix Priority 4)
                f"{question} unlikely challenges obstacles opposing view",
            ]

        elif qt == QuestionType.DEADLINE_EVENT:
            queries = [
                # 1. Direct question
                question,
                # 2. Recent official statements
                f"{question} official statement announcement 2026",
                # 3. OPPOSING VIEW — why it might NOT happen (Fix Priority 4)
                # Gives LLM balanced input to prevent one-sided bias
                f"{question} obstacles challenges unlikely why not",
            ]

        elif qt == QuestionType.HEAD_TO_HEAD:
            queries = [
                # 1. Direct match
                question,
                # 2. Recent form + head-to-head history
                f"{question} head to head recent form 2026",
                # 3. Injuries / lineup / conditions
                f"{question} injury report lineup prediction",
            ]
        else:
            queries = self._generic_queries(market)

        return queries[:3]  # Max 3 queries per market

    def _generic_queries(self, market: Market) -> list[str]:
        """Generic fallback queries when no classification is available."""
        question = market.question
        queries = [
            question,
            f"{question} analysis prediction forecast 2026",
        ]
        cat = (market.category or "").lower()
        if "politic" in cat or "election" in cat:
            queries.append(f"{question} polls latest news")
        elif "crypto" in cat:
            queries.append(f"{question} crypto market analysis")
        elif "econ" in cat:
            queries.append(f"{question} economic indicators data")
        else:
            queries.append(f"{question} expert opinion analysis")
        return queries[:3]
