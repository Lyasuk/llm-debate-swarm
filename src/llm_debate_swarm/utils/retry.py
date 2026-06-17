"""Exponential backoff retry decorator for API calls."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from llm_debate_swarm.utils.logger import get_logger

log = get_logger("retry")

F = TypeVar("F", bound=Callable[..., Any])


def retry_async(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
):
    """Async retry decorator with exponential backoff and jitter."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        log.error(
                            f"[retry] {func.__name__} failed after {max_retries + 1} attempts: {exc}"
                        )
                        raise
                    delay = min(base_delay * (2**attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.2)
                    wait = delay + jitter
                    log.warning(
                        f"[retry] {func.__name__} attempt {attempt + 1} failed: {exc}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    await asyncio.sleep(wait)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


class RateLimiter:
    """Simple token bucket rate limiter for async contexts."""

    def __init__(self, calls_per_second: float = 10.0):
        self._rate = calls_per_second
        self._tokens = calls_per_second
        self._last_refill = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_refill
            self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens < 1:
                wait = (1 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._tokens = 0
            else:
                self._tokens -= 1
