"""
Circuit Breaker & Retry Resilience — FR-67
Tenacity-based retry wrappers with exponential back-off for:
  - Neo4j graph queries
  - Groq LLM calls
  - FAISS vector store operations

Usage:
    from src.resilience.circuit_breaker import with_neo4j_resilience

    @with_neo4j_resilience
    def my_neo4j_call():
        ...
"""

import logging
from functools import wraps

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError,
)

logger = logging.getLogger(__name__)

# ── Shared retry logger ─────────────────────────────────────────
_before_sleep = before_sleep_log(logger, logging.WARNING)


# ── Neo4j Resilience ─────────────────────────────────────────────
def with_neo4j_resilience(func):
    """
    Decorator: retry Neo4j calls up to 3 times with exponential back-off.
    Waits 2s, 4s, 8s between attempts. Logs every retry.
    On exhaustion raises the last exception (caller should handle it).
    """
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=_before_sleep,
        reraise=True,
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


# ── Groq LLM Resilience ──────────────────────────────────────────
def with_groq_resilience(func):
    """
    Decorator: retry Groq LLM calls up to 3 times.
    Waits 1s, 2s, 4s — handles transient rate limits and timeouts.
    """
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        before_sleep=_before_sleep,
        reraise=True,
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


# ── FAISS Resilience ─────────────────────────────────────────────
def with_faiss_resilience(func):
    """
    Decorator: retry FAISS operations up to 2 times.
    Short back-off since FAISS is in-process — retries handle transient index locks.
    """
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
        retry=retry_if_exception_type(Exception),
        before_sleep=_before_sleep,
        reraise=True,
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


# ── Graceful fallback helper ─────────────────────────────────────
def safe_call(func, fallback, *args, **kwargs):
    """
    Call func(*args, **kwargs); return fallback value on any exception.
    Use this when you want to suppress retries and just continue.

    Example:
        result = safe_call(graph_retriever.query, default_result, nl_query=q)
    """
    try:
        return func(*args, **kwargs)
    except (RetryError, Exception) as e:
        logger.warning(f"safe_call fallback triggered for {func.__name__}: {e}")
        return fallback

async def async_safe_call(func, fallback, *args, **kwargs):
    """
    Call await func(*args, **kwargs); return fallback value on any exception.
    """
    try:
        return await func(*args, **kwargs)
    except (RetryError, Exception) as e:
        logger.warning(f"async_safe_call fallback triggered for {func.__name__}: {e}")
        return fallback

