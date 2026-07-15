"""
Agent Tools — Phase 4
Five LangGraph-compatible tools for the KG-RAG agent:
1. vector_search — FAISS semantic similarity search
2. graph_query — Neo4j knowledge graph traversal
3. entity_extraction — Extract entities from query text
4. code_search — Code-specific vector search
5. source_verification — Verify claims against retrieved chunks
"""

import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AgentTools:
    """
    Container for all agent tool functions.
    Injected with vector_store, graph_retriever, embedder, and groq_client instances.
    """

    def __init__(
        self,
        vector_store,
        graph_retriever,
        embedder,
        groq_client,
        llm_model: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.graph_retriever = graph_retriever
        self.embedder = embedder
        self.groq_client = groq_client
        self.llm_model = llm_model or os.getenv("LLM_QUERY_MODEL", "llama-3.3-70b-versatile")

    # ── Tool 1: Vector Search ──────────────────────────────────

    def vector_search(
        self,
        query: str,
        top_k: int = 10,
        user_role: str = "user",
        metadata_filter: Optional[dict] = None,
    ) -> dict:
        """
        Semantic similarity search against the FAISS vector store.

        Returns:
            dict with results list and metadata.
        """
        start = time.time()
        try:
            query_embedding = self.embedder.embed_query(query)
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                user_role=user_role,
                metadata_filter=metadata_filter,
            )
            duration = (time.time() - start) * 1000
            logger.info(f"vector_search: {len(results)} results in {duration:.0f}ms")
            return {
                "tool": "vector_search",
                "success": True,
                "results": [
                    {
                        "chunk_id": r.chunk_id,
                        "doc_id": r.doc_id,
                        "filename": r.filename,
                        "content": r.content,
                        "score": r.score,
                        "rank": r.rank,
                        "metadata": r.metadata,
                    }
                    for r in results
                ],
                "result_count": len(results),
                "duration_ms": duration,
            }
        except Exception as e:
            logger.error(f"vector_search failed: {e}")
            return {"tool": "vector_search", "success": False, "results": [], "error": str(e)}

    # ── Tool 2: Graph Query ─────────────────────────────────────

    def graph_query(
        self,
        nl_query: str,
        hop_depth: int = 3,
        user_role: str = "user",
        start_entity: Optional[str] = None,
        end_entity: Optional[str] = None,
    ) -> dict:
        """
        Knowledge graph traversal via natural language query.

        Returns:
            dict with entities, paths, and Cypher query used.
        """
        start = time.time()
        try:
            if start_entity:
                result = self.graph_retriever.multihop_query(
                    start_entity=start_entity,
                    end_entity=end_entity,
                    hop_depth=hop_depth,
                    user_role=user_role,
                )
            else:
                result = self.graph_retriever.query(
                    nl_query=nl_query,
                    user_role=user_role,
                    hop_depth=hop_depth,
                )

            duration = (time.time() - start) * 1000
            logger.info(
                f"graph_query: {len(result.entities)} entities, "
                f"{len(result.paths)} paths in {duration:.0f}ms"
            )
            return {
                "tool": "graph_query",
                "success": True,
                "entities": result.entities,
                "paths": [
                    {
                        "path_string": p.path_string,
                        "nodes": p.nodes,
                        "relationships": p.relationships,
                    }
                    for p in result.paths
                ],
                "cypher": result.cypher,
                "path_count": len(result.paths),
                "entity_count": len(result.entities),
                "duration_ms": duration,
            }
        except Exception as e:
            logger.error(f"graph_query failed: {e}")
            return {"tool": "graph_query", "success": False, "entities": [], "paths": [], "error": str(e)}

    # ── Tool 3: Entity Extraction ───────────────────────────────

    def entity_extraction(self, text: str) -> dict:
        """
        Extract entities from arbitrary text using Groq LLM.
        Useful for parsing query itself before graph lookup.

        Returns:
            dict with extracted entities list.
        """
        start = time.time()
        try:
            import json
            prompt = f"""Extract named entities from this text. Return JSON array only.
Format: [{{"name": "...", "type": "Person|Team|System|Policy|Incident|Date|Document|CodeModule|Metric"}}]
Text: {text}"""

            response = self.groq_client.chat.completions.create(
                model=os.getenv("LLM_INGEST_MODEL", "llama-3.1-8b-instant"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            entities = json.loads(raw)
            duration = (time.time() - start) * 1000
            return {
                "tool": "entity_extraction",
                "success": True,
                "entities": entities,
                "entity_count": len(entities),
                "duration_ms": duration,
            }
        except Exception as e:
            logger.error(f"entity_extraction failed: {e}")
            return {"tool": "entity_extraction", "success": False, "entities": [], "error": str(e)}

    # ── Tool 4: Code Search ─────────────────────────────────────

    def code_search(
        self,
        query: str,
        top_k: int = 5,
        user_role: str = "user",
    ) -> dict:
        """
        Code-specific search: filters vector results to CodeModule chunks.

        Returns:
            dict with code-specific search results.
        """
        return self.vector_search(
            query=query,
            top_k=top_k,
            user_role=user_role,
            metadata_filter={"file_type": ".py"},  # Code files
        )

    # ── Tool 5: Source Verification ─────────────────────────────

    def source_verification(
        self,
        claim: str,
        source_chunks: list[dict],
    ) -> dict:
        """
        Verify a factual claim against a list of source chunks.
        Returns confidence score and support flag.

        Args:
            claim: The factual claim to verify.
            source_chunks: List of chunk dicts with 'content' field.

        Returns:
            dict with confidence, is_supported, and evidence_snippets.
        """
        start = time.time()
        try:
            if not source_chunks:
                return {
                    "tool": "source_verification",
                    "success": True,
                    "is_supported": False,
                    "confidence": 0.0,
                    "evidence_snippets": [],
                    "unsupported_claim": True,
                }

            context = "\n\n---\n\n".join(
                f"[Source {i+1}: {c.get('filename', 'unknown')}]\n{c.get('content', '')[:500]}"
                for i, c in enumerate(source_chunks[:5])
            )

            prompt = f"""You are a fact-verification engine. Determine if the CLAIM is supported by the SOURCES.

CLAIM: {claim}

SOURCES:
{context}

Respond with JSON only:
{{"is_supported": true/false, "confidence": 0.0-1.0, "evidence": "quote from source that supports/refutes"}}"""

            response = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            import json
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            result = json.loads(raw)
            duration = (time.time() - start) * 1000

            return {
                "tool": "source_verification",
                "success": True,
                "is_supported": result.get("is_supported", False),
                "confidence": float(result.get("confidence", 0.0)),
                "evidence_snippets": [result.get("evidence", "")],
                "duration_ms": duration,
            }
        except Exception as e:
            logger.error(f"source_verification failed: {e}")
            return {
                "tool": "source_verification",
                "success": False,
                "is_supported": False,
                "confidence": 0.0,
                "error": str(e),
            }
