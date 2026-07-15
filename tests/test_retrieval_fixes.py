import pytest
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridRetriever, HybridResult
from src.retrieval.cache import QueryCache


def test_inject_rbac():
    retriever = GraphRetriever()
    cypher1 = "MATCH (n) RETURN n"
    injected1 = retriever._inject_rbac(cypher1, "user1")
    assert "WITH * WHERE" in injected1
    assert injected1.count("WITH * WHERE") == 1
    assert "RETURN" in injected1
    
    # Test multi-line and existing WITH
    cypher2 = "MATCH (n)\nWITH n\nRETURN n"
    injected2 = retriever._inject_rbac(cypher2, "user2")
    assert injected2.count("WITH * WHERE") == 1
    
    # Test double inject prevention
    injected3 = retriever._inject_rbac(injected1, "user3")
    assert injected3.count("WITH * WHERE") == 1


def test_hybrid_retriever_dynamic_score():
    retriever = HybridRetriever(vector_weight=0.6, graph_weight=0.4)
    v_results = [{"chunk_id": "c1", "score": 0.8, "doc_id": "d1", "content": "foo"}]
    
    # Base graph score is 0.5 + 0.1 * hits
    # Here, chunk c1 is hit exactly once. Score = 0.6
    g_results = [{"source_chunk_ids": ["c1"]}]
    
    merged = retriever.merge(v_results, g_results)
    assert len(merged) == 1
    # Check that graph_score was computed dynamically as 0.6
    assert merged[0].graph_score == 0.6
    
    # 6 hits, capped at 0.95
    g_results2 = [{"source_chunk_ids": ["c1", "c1", "c1", "c1", "c1", "c1"]}]
    merged2 = retriever.merge(v_results, g_results2)
    assert merged2[0].graph_score == 0.95


@pytest.mark.asyncio
async def test_cache_reconnect():
    cache = QueryCache(redis_url="redis://localhost:6379/0", ttl_seconds=3600)
    # Simulate a cache that was previously connected but is now unavailable
    cache._client = "dummy_dead_client"
    cache._available = False
    
    # Call _get_client() should trigger reconnection (reset _client to None and attempt new connection)
    # Assuming redis server isn't running on the exact test port, it should fail gracefully 
    # but the crucial buggy behavior was returning the dead client itself.
    client = await cache._get_client()
    assert client is None
    assert cache.is_available is False
