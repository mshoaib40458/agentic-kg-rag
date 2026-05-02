"""
Prometheus Metrics — NFR / §2.4
Defines all application-level counters and histograms.
Exposes /metrics scrape endpoint via FastAPI (no Docker required).

Metrics:
  kgrag_query_total              — counter (query_type, user_role, status)
  kgrag_query_duration_seconds   — histogram (query_type)
  kgrag_ingestion_total          — counter (status)
  kgrag_ingestion_duration_seconds — histogram
  kgrag_agent_iterations_total   — counter
  kgrag_cache_hits_total         — counter
  kgrag_cache_misses_total       — counter
  kgrag_graph_query_duration_seconds — histogram (hop_depth)
  kgrag_llm_calls_total          — counter (model, status)
"""

from prometheus_client import Counter, Histogram, REGISTRY

# ── Query Metrics ──────────────────────────────────────────────
QUERY_TOTAL = Counter(
    "kgrag_query_total",
    "Total number of queries processed",
    ["query_type", "user_role", "status"],  # status: success | error | cached
)

QUERY_DURATION = Histogram(
    "kgrag_query_duration_seconds",
    "End-to-end query latency in seconds",
    ["query_type"],
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0],
)

# ── Ingestion Metrics ──────────────────────────────────────────
INGESTION_TOTAL = Counter(
    "kgrag_ingestion_total",
    "Total documents ingested",
    ["status"],  # success | failure
)

INGESTION_DURATION = Histogram(
    "kgrag_ingestion_duration_seconds",
    "Document ingestion duration in seconds",
    buckets=[1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0],
)

# ── Agent Metrics ──────────────────────────────────────────────
AGENT_ITERATIONS = Counter(
    "kgrag_agent_iterations_total",
    "Total agent re-planning iterations",
)

# ── Cache Metrics ──────────────────────────────────────────────
CACHE_HITS = Counter(
    "kgrag_cache_hits_total",
    "Total cache hits",
)

CACHE_MISSES = Counter(
    "kgrag_cache_misses_total",
    "Total cache misses",
)

# ── Graph Metrics ──────────────────────────────────────────────
GRAPH_QUERY_DURATION = Histogram(
    "kgrag_graph_query_duration_seconds",
    "Knowledge graph query latency in seconds",
    ["hop_depth"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

# ── LLM Metrics ───────────────────────────────────────────────
LLM_CALLS = Counter(
    "kgrag_llm_calls_total",
    "Total LLM API calls made",
    ["model", "status"],  # status: success | error
)


# ── Helper functions ───────────────────────────────────────────

def record_query(
    query_type: str,
    user_role: str,
    status: str,
    duration_seconds: float,
):
    """Record a completed query."""
    QUERY_TOTAL.labels(
        query_type=query_type or "unknown",
        user_role=user_role or "unknown",
        status=status,
    ).inc()
    QUERY_DURATION.labels(query_type=query_type or "unknown").observe(duration_seconds)


def record_ingestion(status: str, duration_seconds: float):
    """Record a completed ingestion."""
    INGESTION_TOTAL.labels(status=status).inc()
    INGESTION_DURATION.observe(duration_seconds)


def record_agent_iteration():
    """Record one agent re-planning iteration."""
    AGENT_ITERATIONS.inc()


def record_cache_hit():
    CACHE_HITS.inc()


def record_cache_miss():
    CACHE_MISSES.inc()


def record_graph_query(hop_depth: int, duration_seconds: float):
    """Record a graph query with its hop depth."""
    GRAPH_QUERY_DURATION.labels(hop_depth=str(hop_depth)).observe(duration_seconds)


def record_llm_call(model: str, status: str = "success"):
    """Record an LLM API call."""
    LLM_CALLS.labels(model=model, status=status).inc()
