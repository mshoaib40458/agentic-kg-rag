"""
Admin Routes — Phase 9 (Production)
GET  /api/admin/stats          — system statistics
GET  /api/admin/audit          — paginated audit log
POST /api/admin/users          — create users
DELETE /api/admin/documents/{doc_id} — remove document from FAISS + Neo4j
POST /api/admin/cache/flush    — flush query result cache
"""

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.security.auth import TokenData, require_admin, create_user
from src.audit.logger import audit_logger
from src.retrieval.cache import query_cache

logger = logging.getLogger(__name__)
router = APIRouter()


class SystemStats(BaseModel):
    vector_store: dict
    graph_store: dict
    audit_log_size: int
    cache_available: bool


class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


@router.get("/stats", response_model=SystemStats)
async def get_stats(current_user: TokenData = Depends(require_admin)):
    """Return system statistics. Admin only."""
    from src.api.main import app_state
    vs = app_state.get("vector_store")
    gr = app_state.get("graph_retriever")

    vs_stats = vs.get_stats() if vs else {}
    if not isinstance(vs_stats, dict):
        vs_stats = {"mocked": True}
    try:
        from src.ingestion.kg_builder import KGBuilder
        kg = KGBuilder()
        kg.connect()
        gs_stats = kg.get_stats()
        kg.disconnect()
    except Exception:
        gs_stats = {"error": "Neo4j not reachable"}

    # Count actual log entries (JSONL line count)
    audit_log_size = 0
    try:
        from src.audit.logger import LOG_PATH
        if LOG_PATH.exists():
            with open(LOG_PATH, "r", encoding="utf-8") as _f:
                audit_log_size = sum(1 for line in _f if line.strip())
    except Exception:
        audit_log_size = -1  # Unavailable

    return SystemStats(
        vector_store=vs_stats,
        graph_store=gs_stats,
        audit_log_size=audit_log_size,
        cache_available=await query_cache.check_available(),
    )


@router.get("/audit")
async def get_audit_logs(
    limit: int = Query(default=50, le=500),
    event_type: Optional[str] = Query(default=None),
    user_id: Optional[str] = Query(default=None),
    current_user: TokenData = Depends(require_admin),
):
    """Return paginated audit logs. Admin only."""
    records = audit_logger.read_logs(
        limit=limit,
        event_type=event_type,
        user_id=user_id,
    )
    return {"records": records, "count": len(records)}


@router.post("/users")
async def create_new_user(
    request: UserCreateRequest,
    current_user: TokenData = Depends(require_admin),
):
    """Create a new user. Admin only."""
    allowed_roles = {"admin", "user", "auditor"}
    if request.role not in allowed_roles:
        raise HTTPException(status_code=400, detail=f"Role must be one of: {allowed_roles}")
    try:
        import asyncio
        loop = asyncio.get_running_loop()
        user = await loop.run_in_executor(None, create_user, request.username, request.password, request.role)
        audit_logger.log_query(
            user_id=current_user.user_id,
            user_role=current_user.role,
            session_id="admin",
            query=f"Created user: {request.username} role={request.role}",
        )
        return {"status": "created", **user}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: TokenData = Depends(require_admin),
):
    """
    Remove a document from the vector store and knowledge graph. Admin only.
    Returns counts of removed chunks and graph entities.
    """
    from src.api.main import app_state
    vs = app_state.get("vector_store")
    gr = app_state.get("graph_retriever")

    # 1. Remove from FAISS
    chunks_removed = 0
    if vs:
        try:
            chunks_removed = vs.delete_by_doc_id(doc_id)
        except Exception as e:
            logger.error(f"FAISS delete failed for doc_id={doc_id}: {e}")
            raise HTTPException(status_code=500, detail=f"FAISS delete failed: {e}")

    # 2. Remove from Neo4j — two safe queries instead of APOC-dependent CALL{UNION}
    entities_removed = 0
    if gr:
        try:
            driver = gr._get_driver()
            with driver.session() as session:
                # Query 1: Fully delete entities that ONLY belong to this doc
                result1 = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE $doc_id IN e.source_doc_ids
                      AND size(e.source_doc_ids) = 1
                    WITH count(e) AS cnt
                    MATCH (e:Entity)
                    WHERE $doc_id IN e.source_doc_ids
                      AND size(e.source_doc_ids) = 1
                    DETACH DELETE e
                    RETURN cnt
                    """,
                    doc_id=doc_id,
                )
                rec1 = result1.single()
                entities_removed += rec1["cnt"] if rec1 else 0

                # Query 2: Update entities shared with other docs — just remove this doc_id
                session.run(
                    """
                    MATCH (e:Entity)
                    WHERE $doc_id IN e.source_doc_ids
                      AND size(e.source_doc_ids) > 1
                    SET e.source_doc_ids = [x IN e.source_doc_ids WHERE x <> $doc_id]
                    """,
                    doc_id=doc_id,
                )
        except Exception as e:
            logger.warning(f"Neo4j delete failed for doc_id={doc_id}: {e}")
            entities_removed = -1  # Indicate partial failure

    # 3. Flush related cache entries (flush all since we don't track per-doc)
    await query_cache.flush_all()

    audit_logger.log_ingestion(
        user_id=current_user.user_id,
        filename=f"[DELETED] doc_id={doc_id}",
        doc_id=doc_id,
        chunk_count=-chunks_removed,
        entity_count=-entities_removed,
        relationship_count=0,
        duration_seconds=0.0,
    )

    return {
        "status": "deleted",
        "doc_id": doc_id,
        "chunks_removed": chunks_removed,
        "entities_removed": entities_removed,
        "cache_flushed": True,
    }


@router.post("/cache/flush")
async def flush_cache(current_user: TokenData = Depends(require_admin)):
    """Flush all query result cache entries. Admin only."""
    success = await query_cache.flush_all()
    return {"status": "flushed" if success else "cache_unavailable"}
