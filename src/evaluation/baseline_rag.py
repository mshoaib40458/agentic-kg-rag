"""
Baseline RAG — FR-56
Pure vector-only Retrieval-Augmented Generation — NO agent orchestration.
Used as the comparison baseline for the 40% accuracy improvement target.

Architecture:
  1. Embed query with sentence-transformers
  2. FAISS similarity search (top-k)
  3. Direct Groq LLM call with retrieved context
  4. Return answer + basic citations (no graph, no re-planning)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result from the baseline RAG system."""
    query: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    duration_ms: float = 0.0
    chunk_count: int = 0


BASELINE_PROMPT = """You are an enterprise knowledge assistant.
Answer the following question using ONLY the provided context.
Cite sources inline as [Source: filename].
If the answer is not in the context, say "Information not available in knowledge base."

Question: {query}

Context:
{context}

Answer:"""


class BaselineRAG:
    """
    Traditional vector-only RAG system.
    Retrieves top-k chunks from FAISS, generates answer directly with LLM.
    No agent, no graph, no re-planning.
    """

    def __init__(
        self,
        vector_store=None,
        embedder=None,
        groq_api_key: Optional[str] = None,
        llm_model: str = "llama-3.1-8b-instant",
        top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_model = llm_model
        self.top_k = top_k
        self._groq = None
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if api_key:
            from groq import Groq
            self._groq = Groq(api_key=api_key)

    def run(
        self,
        query: str,
        user_role: str = "user",
        top_k: Optional[int] = None,
    ) -> BaselineResult:
        """
        Run baseline RAG for a query.

        Args:
            query: Natural language query.
            user_role: RBAC role for retrieval filtering.
            top_k: Number of chunks to retrieve.

        Returns:
            BaselineResult with answer and citations.
        """
        start = time.time()
        k = top_k or self.top_k

        # ── Step 1: Embed query ──────────────────────────────
        if not self.vector_store or not self.embedder:
            return BaselineResult(
                query=query,
                answer="Baseline RAG: vector store or embedder not initialized.",
                duration_ms=0.0,
            )

        try:
            query_embedding = self.embedder.embed_query(query)
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=k,
                user_role=user_role,
            )
        except Exception as e:
            logger.error(f"Baseline retrieval failed: {e}")
            return BaselineResult(
                query=query,
                answer=f"Retrieval failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

        # ── Step 2: Build context ─────────────────────────────
        context_parts = []
        citations = []
        for i, r in enumerate(results):
            filename = getattr(r, "filename", r.get("filename", "unknown") if isinstance(r, dict) else "unknown")
            content = getattr(r, "content", r.get("content", "") if isinstance(r, dict) else "")
            chunk_id = getattr(r, "chunk_id", r.get("chunk_id", "") if isinstance(r, dict) else "")
            score = getattr(r, "score", r.get("score", 0.0) if isinstance(r, dict) else 0.0)

            context_parts.append(f"[Source {i+1}: {filename}]\n{content[:600]}")
            citations.append({"filename": filename, "chunk_id": chunk_id, "score": float(score)})

        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

        # ── Step 3: Generate answer ───────────────────────────
        if not self._groq:
            answer = f"[No LLM configured] Context retrieved: {len(results)} chunks"
            confidence = 0.5 if results else 0.0
        else:
            try:
                prompt = BASELINE_PROMPT.format(query=query, context=context)
                response = self._groq.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1024,
                )
                answer = response.choices[0].message.content.strip()
                # Heuristic confidence: average retrieval score
                scores = [c["score"] for c in citations if c["score"] > 0]
                confidence = round(sum(scores) / len(scores), 3) if scores else 0.5
            except Exception as e:
                logger.error(f"Baseline LLM call failed: {e}")
                answer = f"LLM generation failed: {e}"
                confidence = 0.0

        duration_ms = (time.time() - start) * 1000
        logger.info(
            f"Baseline RAG: query='{query[:60]}' "
            f"chunks={len(results)} confidence={confidence:.2f} "
            f"duration={duration_ms:.0f}ms"
        )

        return BaselineResult(
            query=query,
            answer=answer,
            citations=citations,
            confidence=confidence,
            duration_ms=duration_ms,
            chunk_count=len(results),
        )
