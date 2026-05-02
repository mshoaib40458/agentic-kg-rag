"""
VectorStore Tests — add, search, RBAC filtering, delete, save/load.
SearchResult objects are dataclasses — use attribute access (result.chunk_id),
not dict access (result["chunk_id"]).
"""

import numpy as np
import pytest


def _ids(results) -> list:
    """Helper: extract chunk_ids from SearchResult dataclasses OR dicts."""
    return [r.chunk_id if hasattr(r, "chunk_id") else r["chunk_id"] for r in results]


def test_add_and_search(temp_vector_store, sample_chunks, sample_embeddings):
    """Adding chunks and searching should return relevant results."""
    temp_vector_store.add_no_save(sample_chunks, sample_embeddings)

    results = temp_vector_store.search(
        query_embedding=sample_embeddings[0],
        top_k=3,
        user_role="admin",
    )
    assert len(results) >= 1
    assert _ids(results)[0] == "chunk-001"


def test_add_batch_and_save(tmp_path, sample_chunks, sample_embeddings):
    """add_no_save + save() should persist the index to disk."""
    index_path = tmp_path / "idx.bin"
    meta_path = tmp_path / "meta.json"
    from src.retrieval.vector_store import VectorStore
    vs = VectorStore(str(index_path), str(meta_path), embedding_dim=384)
    vs.add_no_save(sample_chunks, sample_embeddings)
    vs.save()
    assert index_path.with_suffix(".bin").exists() or index_path.exists()
    assert meta_path.exists()


def test_load_persisted_store(tmp_path, sample_chunks, sample_embeddings):
    """VectorStore should reload saved data from disk."""
    index_path = tmp_path / "reload"
    meta_path = tmp_path / "reload.json"
    from src.retrieval.vector_store import VectorStore
    vs1 = VectorStore(str(index_path), str(meta_path), embedding_dim=384)
    vs1.add(sample_chunks, sample_embeddings)  # add() calls save()

    # Load fresh instance
    vs2 = VectorStore(str(index_path), str(meta_path), embedding_dim=384)
    vs2.load()
    stats = vs2.get_stats()
    assert stats["total_vectors"] == 3


def test_rbac_filtering(temp_vector_store, sample_chunks, sample_embeddings):
    """RBAC role filtering should exclude chunks not accessible to the user."""
    from dataclasses import dataclass, field

    @dataclass
    class RestrictedChunk:
        chunk_id: str = "chunk-restricted"
        doc_id: str = "doc-secret"
        filename: str = "secret.txt"
        content: str = "Top secret content"
        chunk_index: int = 0
        total_chunks: int = 1
        char_start: int = 0
        char_end: int = 50
        access_roles: list = field(default_factory=lambda: ["admin"])  # admin only
        embedding_model_id: str = "all-MiniLM-L6-v2"
        embedding_version: str = "v1"
        metadata: dict = field(default_factory=dict)

    rng = np.random.default_rng(99)
    restricted_emb = rng.random((1, 384)).astype(np.float32)
    restricted_emb /= np.linalg.norm(restricted_emb, axis=1, keepdims=True)

    temp_vector_store.add_no_save(sample_chunks, sample_embeddings)
    temp_vector_store.add_no_save([RestrictedChunk()], restricted_emb)

    # Search as "user" role — should not see admin-only chunk
    user_results = temp_vector_store.search(
        query_embedding=restricted_emb[0],
        top_k=5,
        user_role="user",
    )
    assert "chunk-restricted" not in _ids(user_results)

    # Search as "admin" role — should see it
    admin_results = temp_vector_store.search(
        query_embedding=restricted_emb[0],
        top_k=5,
        user_role="admin",
    )
    assert "chunk-restricted" in _ids(admin_results)


def test_delete_by_doc_id(temp_vector_store, sample_chunks, sample_embeddings):
    """delete_by_doc_id should remove only the matching doc's chunks."""
    temp_vector_store.add_no_save(sample_chunks, sample_embeddings)

    removed = temp_vector_store.delete_by_doc_id("doc-abc")
    assert removed == 2  # chunk-001 and chunk-002 belong to doc-abc

    stats = temp_vector_store.get_stats()
    assert stats["total_vectors"] == 1  # chunk-003 from doc-xyz remains


def test_delete_nonexistent_doc(temp_vector_store, sample_chunks, sample_embeddings):
    """delete_by_doc_id on a missing doc_id returns 0 without error."""
    temp_vector_store.add_no_save(sample_chunks, sample_embeddings)
    removed = temp_vector_store.delete_by_doc_id("nonexistent-doc")
    assert removed == 0


def test_empty_store_returns_empty_search(temp_vector_store, sample_embeddings):
    """Searching an empty store returns empty list, not an error."""
    results = temp_vector_store.search(
        query_embedding=sample_embeddings[0],
        top_k=5,
        user_role="admin",
    )
    assert results == []
