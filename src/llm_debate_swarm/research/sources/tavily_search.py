"""Tavily web search integration with caching + quota fallback."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from llm_debate_swarm.config import get_api_key
from llm_debate_swarm.utils.logger import get_logger
from llm_debate_swarm.utils.retry import retry_async

log = get_logger("research.tavily")


@dataclass
class SearchResult:
    """A single search result."""

    title: str = ""
    url: str = ""
    content: str = ""
    score: float = 0.0


@dataclass
class TavilySearchOutput:
    """Aggregated output from Tavily search."""

    query: str = ""
    results: list[SearchResult] = field(default_factory=list)
    answer: str = ""  # Tavily's AI-generated summary
    cached: bool = False  # чи результат з кешу
    fallback: str = ""  # "duckduckgo" якщо fallback використано


# Cache config
CACHE_DIR = Path("data/cache/tavily")
CACHE_TTL_HOURS = 24


def _cache_key(query: str) -> str:
    """SHA256 hash of query для file name."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:32]


def _cache_path(query: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{_cache_key(query)}.pkl"


def _load_from_cache(query: str) -> TavilySearchOutput | None:
    """Check if cached result exists and is fresh."""
    path = _cache_path(query)
    if not path.exists():
        return None
    try:
        age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
        if age_hours > CACHE_TTL_HOURS:
            return None
        with open(path, "rb") as f:
            result = pickle.load(f)
        result.cached = True
        log.info(f"Cache HIT ({age_hours:.1f}h old): {query[:50]}...")
        return result
    except Exception as exc:
        log.warning(f"Cache read failed: {exc}")
        return None


def _save_to_cache(query: str, output: TavilySearchOutput) -> None:
    """Save result to cache."""
    try:
        path = _cache_path(query)
        with open(path, "wb") as f:
            # Don't cache the cached flag
            to_save = TavilySearchOutput(
                query=output.query,
                results=output.results,
                answer=output.answer,
                cached=False,
                fallback=output.fallback,
            )
            pickle.dump(to_save, f)
    except Exception as exc:
        log.warning(f"Cache write failed: {exc}")


async def _duckduckgo_fallback(query: str) -> TavilySearchOutput:
    """Fallback на DuckDuckGo коли Tavily quota вичерпано."""
    try:
        import httpx
        # DDG HTML results (no API key, no quota)
        url = "https://html.duckduckgo.com/html/"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                url,
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (compatible; poly-think/1.0)"},
            )
        # Very basic HTML parsing — extract result snippets
        html = resp.text
        results = []
        # Simple regex-based extraction
        import re
        # DDG result pattern
        pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>'
            r'.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        matches = list(pattern.finditer(html))[:10]
        for m in matches:
            url_, title, snippet = m.group(1), m.group(2), m.group(3)
            snippet_clean = re.sub(r"<[^>]+>", "", snippet).strip()[:500]
            results.append(SearchResult(
                title=title.strip()[:200],
                url=url_,
                content=snippet_clean,
                score=0.5,
            ))
        log.info(f"DuckDuckGo fallback: {len(results)} results for {query[:50]}")
        return TavilySearchOutput(
            query=query,
            results=results,
            answer="",
            cached=False,
            fallback="duckduckgo",
        )
    except Exception as exc:
        log.warning(f"DuckDuckGo fallback failed: {exc}")
        return TavilySearchOutput(query=query, fallback="failed")


class TavilySearch:
    """Tavily AI-native search API with caching + fallback."""

    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self._client = None
        self._quota_exhausted = False  # once True, skip to fallback

    async def _get_client(self):
        if self._client is None:
            from tavily import AsyncTavilyClient
            api_key = get_api_key("TAVILY_API_KEY")
            self._client = AsyncTavilyClient(api_key=api_key)
        return self._client

    async def search(self, query: str) -> TavilySearchOutput:
        """Run search with cache + fallback."""
        # 1. Check cache first
        cached = _load_from_cache(query)
        if cached is not None:
            return cached

        # 2. If quota known to be exhausted, go straight to fallback
        if self._quota_exhausted:
            log.info(f"Quota exhausted, using DDG: {query[:50]}")
            return await _duckduckgo_fallback(query)

        # 3. Try Tavily
        log.info(f"Searching: {query[:80]}...")
        try:
            result = await self._tavily_search(query)
            # Cache successful result
            if result.results:
                _save_to_cache(query, result)
            return result
        except Exception as exc:
            exc_str = str(exc).lower()
            if "usage limit" in exc_str or "plan" in exc_str or "429" in exc_str or "quota" in exc_str:
                log.error(f"Tavily quota exhausted, switching to DDG fallback")
                self._quota_exhausted = True
                return await _duckduckgo_fallback(query)
            log.warning(f"Search failed: {exc}")
            raise

    @retry_async(max_retries=2, base_delay=2.0)
    async def _tavily_search(self, query: str) -> TavilySearchOutput:
        """Actual Tavily API call with retry."""
        client = await self._get_client()
        raw = await client.search(
            query=query,
            search_depth="advanced",
            max_results=self.max_results,
            include_answer=True,
        )

        results = []
        for item in raw.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=item.get("score", 0.0),
            ))

        output = TavilySearchOutput(
            query=query,
            results=results,
            answer=raw.get("answer", ""),
        )

        log.info(f"Got {len(results)} results for: {query[:50]}...")
        return output

    async def multi_search(self, queries: list[str]) -> list[TavilySearchOutput]:
        """Run multiple searches sequentially (Tavily rate limits)."""
        import asyncio
        results = []
        for query in queries:
            try:
                result = await self.search(query)
                results.append(result)
                await asyncio.sleep(0.5)  # gentle rate limiting
            except Exception as exc:
                log.warning(f"Search failed for '{query[:50]}': {exc}")
                results.append(TavilySearchOutput(query=query))
        return results
