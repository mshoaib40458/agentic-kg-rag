"""
FAISS Vector Store — Phase 2
Persistent FAISS index with JSON metadata store.
Supports RBAC-filtered similarity search and embedding versioning.
"""

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single vector search result."""
    chunk_id: str
    doc_id: str
    filename: str
    content: str
    score: float
    metadata: dict
    rank: int = 0


class VectorStore:
    """
    FAISS-based vector store with metadata sidecar (JSON).
    Supports filtering by RBAC access_roles and metadata fields.
    Handles embedding versioning for safe index migration.
    """

    def __init__(
        self,
        index_path: str = "data/faiss_index",
        metadata_path: str = "data/faiss_metadata.json",
        embedding_dim: int = 384,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_dim = embedding_dim

        self._index: Optional[faiss.IndexFlatIP] = None  # Inner product = cosine for normalized vectors
        self._metadata: list[dict] = []  # Parallel metadata list (index i ↔ metadata[i])
        self._lock = threading.RLock()  # Prevent concurrent modification during deletion

        # Ensure directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Index Management ───────────────────────────────────────

    def _get_or_create_index(self) -> faiss.IndexFlatIP:
        if self._index is None:
            if self.index_path.with_suffix(".bin").exists():
                self.load()
            else:
                self._index = faiss.IndexFlatIP(self.embedding_dim)
                self._metadata = []
                logger.info(f"Created new FAISS IndexFlatIP (dim={self.embedding_dim})")
        return self._index

    def save(self):
        """Persist FAISS index and metadata to disk."""
        faiss.write_index(self._index, str(self.index_path.with_suffix(".bin")))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved FAISS index ({self._index.ntotal} vectors) to {self.index_path}")

    def load(self):
        """Load FAISS index and metadata from disk."""
        index_file = self.index_path.with_suffix(".bin")
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")

        self._index = faiss.read_index(str(index_file))

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

        logger.info(f"✓ Loaded FAISS index ({self._index.ntotal} vectors) from {self.index_path}")

    # ── Write Operations ───────────────────────────────────────

    def add(self, chunks: list, embeddings: np.ndarray):
        """
        Add chunks and their embeddings to the store, then save.

        Args:
            chunks: List of DocumentChunk objects.
            embeddings: numpy array (N, dim) of normalized embeddings.
        """
        self.add_no_save(chunks, embeddings)
        self.save()

    def add_no_save(self, chunks: list, embeddings: np.ndarray):
        """
        Add chunks and embeddings to in-memory index WITHOUT saving to disk.
        Call save() explicitly after batch operations for best performance.

        Args:
            chunks: List of DocumentChunk objects.
            embeddings: numpy array (N, dim) of normalized embeddings.
        """
        index = self._get_or_create_index()

        if len(chunks) == 0:
            return

        # Ensure float32
        embeddings = embeddings.astype(np.float32)

        # Add to FAISS
        index.add(embeddings)

        # Add parallel metadata
        for chunk in chunks:
            self._metadata.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "access_roles": chunk.access_roles,
                "embedding_model_id": chunk.embedding_model_id,
                "embedding_version": chunk.embedding_version,
                "metadata": chunk.metadata,
            })

        logger.info(f"Added {len(chunks)} chunks to FAISS in-memory index. Total: {index.ntotal}")

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Remove all chunks for a given doc_id and rebuild the FAISS index.
        FAISS-CPU flat index does not support in-place deletion, so we rebuild.
        Uses locking to prevent concurrent modification.

        Args:
            doc_id: Document ID to remove.

        Returns:
            Number of chunks removed.
        """
        with self._lock:  # Prevent concurrent access during rebuild
            if self._index is None or self._index.ntotal == 0:
                return 0

            # Find which positions to keep
            keep_indices = [i for i, m in enumerate(self._metadata) if m.get("doc_id") != doc_id]
            removed_count = len(self._metadata) - len(keep_indices)

            if removed_count == 0:
                return 0

            # Rebuild metadata list
            new_metadata = [self._metadata[i] for i in keep_indices]

            # Rebuild FAISS index by extracting kept vectors
            # faiss.IndexFlatIP supports reconstruct(i) for vector extraction
            new_index = faiss.IndexFlatIP(self.embedding_dim)
            
            failed_reconstructions = 0
            for i in keep_indices:
                try:
                    vec = self._index.reconstruct(i)
                    new_index.add(vec.reshape(1, -1))
                except Exception as e:
                    logger.warning(f"Could not reconstruct vector {i}: {e}")
                    failed_reconstructions += 1

            if failed_reconstructions > 0:
                logger.warning(f"Failed to reconstruct {failed_reconstructions}/{len(keep_indices)} vectors")

            # Atomic swap
            self._index = new_index
            self._metadata = new_metadata
            
            try:
                self.save()
            except Exception as e:
                logger.error(f"Failed to save after delete: {e} — index may be inconsistent")
                raise

            logger.info(f"Deleted {removed_count} chunks for doc_id={doc_id}. Remaining: {new_index.ntotal}")
            return removed_count


    # ── Search Operations ──────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        user_role: str = "user",
        metadata_filter: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        RBAC-filtered similarity search with thread-safe locking.

        Args:
            query_embedding: Query embedding vector (dim,).
            top_k: Number of results to return (after RBAC filtering).
            user_role: Current user's role for RBAC filtering.
            metadata_filter: Optional dict of metadata key-value pairs to filter by.

        Returns:
            List of SearchResult objects ranked by similarity.
        """
        with self._lock:  # Prevent concurrent modification during search
            index = self._get_or_create_index()

            if index.ntotal == 0:
                logger.warning("FAISS index is empty — no results.")
                return []

            # Search top (top_k * 5) to allow for RBAC filtering losses
            search_k = min(top_k * 5, index.ntotal)
            query_vec = query_embedding.reshape(1, -1).astype(np.float32)

            scores, indices = index.search(query_vec, search_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx >= len(self._metadata):
                    continue

                meta = self._metadata[idx]

                # RBAC filter
                if user_role not in meta.get("access_roles", []):
                    continue

                # Metadata filter
                if metadata_filter:
                    if not all(
                        str(meta.get("metadata", {}).get(k)) == str(v)
                        for k, v in metadata_filter.items()
                    ):
                        continue

                results.append(SearchResult(
                    chunk_id=meta["chunk_id"],
                    doc_id=meta["doc_id"],
                    filename=meta["filename"],
                    content=meta["content"],
                    score=float(score),
                    metadata=meta.get("metadata", {}),
                ))

                if len(results) >= top_k:
                    break

        # Rank results
        for rank, result in enumerate(results):
            result.rank = rank + 1

        logger.debug(f"Vector search returned {len(results)} results for role='{user_role}'")
        return results

    def get_stats(self) -> dict:
        """Return vector store statistics."""
        if self._index is None:
            return {"total_vectors": 0, "embedding_dim": self.embedding_dim}
        return {
            "total_vectors": self._index.ntotal,
            "embedding_dim": self.embedding_dim,
            "metadata_count": len(self._metadata),
            "index_trained": self._index.is_trained,
        }
