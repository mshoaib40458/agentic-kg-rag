"""
Reranker — Phase 5
Cross-encoder reranking using ms-marco-MiniLM for production-grade relevance scoring.
Falls back to score-based ranking if model unavailable.
"""

import logging
from src.retrieval.hybrid_retriever import HybridResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    Falls back to combined_score ranking if model not loaded.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info(f"✓ Cross-encoder loaded: {self.model_name}")
            except Exception as e:
                logger.warning(f"Cross-encoder load failed (will use score fallback): {e}")
                self._model = None

    def rerank(
        self,
        query: str,
        candidates: list[HybridResult],
        top_k: int = None,
    ) -> list[HybridResult]:
        """
        Rerank candidate chunks for a query using cross-encoder scores.

        Args:
            query: The user's query string.
            candidates: HybridResult objects to rerank.
            top_k: Number of top results to return.

        Returns:
            List of HybridResult, reranked and trimmed to top_k.
        """
        k = top_k or self.top_k

        if not candidates:
            return []

        self._load_model()

        if self._model is None:
            # Fallback: return top_k by combined_score
            logger.info("Reranking fallback: using combined_score order")
            return sorted(candidates, key=lambda x: x.combined_score, reverse=True)[:k]

        try:
            # Build (query, passage) pairs for cross-encoder
            pairs = [(query, c.content) for c in candidates]
            scores = self._model.predict(pairs)

            # Attach cross-encoder scores and re-sort
            for cand, score in zip(candidates, scores):
                cand.combined_score = float(score)

            reranked = sorted(candidates, key=lambda x: x.combined_score, reverse=True)[:k]

            for i, r in enumerate(reranked):
                r.rank = i + 1

            logger.info(f"Reranked {len(candidates)} → top {len(reranked)} results")
            return reranked

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}. Using fallback.")
            return sorted(candidates, key=lambda x: x.combined_score, reverse=True)[:k]
