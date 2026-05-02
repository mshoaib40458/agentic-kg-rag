"""
Hybrid Retriever — Phase 5
Merges and deduplicates results from FAISS vector store and Neo4j graph.
Normalises scores to a unified 0-1 scale for reranking.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """A unified retrieval result from vector + graph sources."""
    chunk_id: str
    doc_id: str
    filename: str
    content: str
    vector_score: float = 0.0
    graph_score: float = 0.0
    combined_score: float = 0.0
    source_type: str = "vector"  # "vector" | "graph" | "hybrid"
    rank: int = 0
    metadata: dict = field(default_factory=dict)


class HybridRetriever:
    """
    Combines vector search and graph retrieval results.
    Deduplicates by chunk_id, normalizes scores, applies weighted fusion.
    """

    def __init__(
        self,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
    ):
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

    def merge(
        self,
        vector_results: list[dict],
        graph_results: list,
        top_k: int = 10,
    ) -> list[HybridResult]:
        """
        Merge and deduplicate vector + graph results.

        Args:
            vector_results: List of vector search result dicts.
            graph_results: List of graph entity/path dicts.
            top_k: Maximum results to return.

        Returns:
            List of HybridResult objects ranked by combined_score.
        """
        merged: dict[str, HybridResult] = {}

        # Helper to extract attributes from either dict or SearchResult dataclass
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                val = obj.get(key, default)
            else:
                val = getattr(obj, key, default)
            if key in ("score", "confidence") and val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default
            return val

        # ── Process vector results ─────────────────────────────
        vector_scores = []
        for r in vector_results:
            score = _get(r, "score", 0.0)
            if score is not None and score >= 0:
                vector_scores.append(score)
                
        max_vec_score = max(vector_scores) if vector_scores else 1.0

        for r in vector_results:
            chunk_id = _get(r, "chunk_id") or _get(r, "doc_id", "")
            if not chunk_id:
                continue
                
            raw_score = _get(r, "score", 0.0) or 0.0
            normalized_score = raw_score / max_vec_score if max_vec_score > 0 else 0.0
            normalized_score = min(1.0, max(0.0, normalized_score))
            
            merged[chunk_id] = HybridResult(
                chunk_id=chunk_id,
                doc_id=_get(r, "doc_id", ""),
                filename=_get(r, "filename", ""),
                content=_get(r, "content", ""),
                vector_score=normalized_score,
                graph_score=0.0,
                source_type="vector",
                metadata=_get(r, "metadata", {}),
            )

        # ── Process graph results ──────────────────────────────
        graph_chunk_ids: dict[str, int] = {}  # chunk_id -> hit count
        if graph_results:
            for item in graph_results:
                if not isinstance(item, dict):
                    continue
                # Graph results may be entity dicts or path dicts
                chunk_ids = item.get("source_chunk_ids", []) or item.get("source_doc_ids", [])
                for cid in chunk_ids:
                    graph_chunk_ids[cid] = graph_chunk_ids.get(cid, 0) + 1

        for cid, hit_count in graph_chunk_ids.items():
            if cid in merged:
                # Dynamic graph score: base 0.5 + 0.1 per additional hit, capped at 0.95
                dynamic_graph_score = min(0.5 + 0.1 * hit_count, 0.95)
                merged[cid].graph_score = dynamic_graph_score
                merged[cid].source_type = "hybrid"

        # ── Compute combined score ─────────────────────────────
        for result in merged.values():
            result.combined_score = round(
                (result.vector_score * self.vector_weight)
                + (result.graph_score * self.graph_weight),
                4,
            )

        # ── Sort and rank ──────────────────────────────────────
        ranked = sorted(merged.values(), key=lambda x: x.combined_score, reverse=True)[:top_k]
        for i, r in enumerate(ranked):
            r.rank = i + 1

        logger.info(
            f"Hybrid merge: {len(vector_results)} vec + {len(graph_results)} graph "
            f"→ {len(ranked)} merged results (top_k={top_k})"
        )
        return ranked
