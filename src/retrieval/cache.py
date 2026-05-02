"""
Query Cache — Phase 3 (Production)
Redis-backed query result cache with role-scoped keys.
Falls back gracefully if Redis is unavailable — no crash, just cache miss.
"""

import hashlib
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class QueryCache:
    """
    Role-scoped Redis cache for agent query results.
    Key format: kgrag:query:{sha256(role + query)}
    Falls back to no-op if Redis is not available.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_seconds: int = 3600,
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.ttl = ttl_seconds
        self._client = None
        self._available = False

    def _get_client(self):
        """Lazy Redis connection. Re-attempts if previously unavailable (self-healing)."""
        # Already connected and healthy
        if self._client is not None and self._available:
            return self._client

        # Either never connected or previously failed — attempt (re-)connection
        # Reset both fields atomically before the attempt
        self._client = None
        self._available = False
        try:
            import redis
            client = redis.Redis.from_url(
                self.redis_url, decode_responses=True, socket_connect_timeout=2
            )
            client.ping()  # Verify connection is live before storing
            self._client = client
            self._available = True
            logger.info(f"✓ Redis cache connected: {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis unavailable — cache disabled: {e}")
            # self._client and self._available already reset above
        return self._client

    def _make_key(self, query: str, user_id: str) -> str:
        """Generate a user-scoped cache key."""
        raw = f"{user_id}:{query.strip().lower()}"
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return f"kgrag:query:{digest}"

    def get(self, query: str, user_id: str) -> Optional[dict]:
        """
        Retrieve cached agent state for a query+user pair.

        Returns:
            Cached state dict if hit, None on miss or error.
        """
        client = self._get_client()
        if not self._available or client is None:
            return None
        try:
            key = self._make_key(query, user_id)
            raw = client.get(key)
            if raw is None:
                return None
            state = json.loads(raw)
            logger.info(f"Cache HIT for query='{query[:60]}' user_id={user_id}")
            return state
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    def set(self, query: str, user_id: str, state: dict) -> bool:
        """
        Cache the agent state for a query+user pair.

        Returns:
            True if cached successfully, False otherwise.
        """
        client = self._get_client()
        if not self._available or client is None:
            return False
        try:
            key = self._make_key(query, user_id)
            # Serialize state — skip non-serializable keys gracefully
            serializable = {k: v for k, v in state.items() if _is_json_serializable(v)}
            client.setex(key, self.ttl, json.dumps(serializable, default=str))
            logger.info(f"Cache SET for query='{query[:60]}' user_id={user_id} ttl={self.ttl}s")
            return True
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
            return False

    def invalidate(self, query: str, user_id: str) -> bool:
        """Delete a specific cache entry."""
        client = self._get_client()
        if not self._available or client is None:
            return False
        try:
            key = self._make_key(query, user_id)
            client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache invalidate failed: {e}")
            return False

    def flush_all(self) -> bool:
        """Flush all KG-RAG cache entries (admin use only)."""
        client = self._get_client()
        if not self._available or client is None:
            return False
        try:
            keys = client.keys("kgrag:query:*")
            if keys:
                client.delete(*keys)
            logger.info(f"Cache flushed: {len(keys)} entries removed")
            return True
        except Exception as e:
            logger.warning(f"Cache flush failed: {e}")
            return False

    @property
    def is_available(self) -> bool:
        self._get_client()
        return self._available


def _is_json_serializable(value) -> bool:
    """Quick check if a value can be JSON serialized without raising."""
    try:
        json.dumps(value, default=str)
        return True
    except Exception:
        return False


# Singleton
query_cache = QueryCache(ttl_seconds=int(os.getenv("REDIS_CACHE_TTL", "3600")))
