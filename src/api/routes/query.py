"""
Query Routes — Phase 9 (Production)
POST /api/query — synchronous agent query
WebSocket /api/stream — streaming agent reasoning + answer

Production Fixes Applied:
- Query max-length validation (2000 chars) via Pydantic Field
- Singleton orchestrator retrieved from app_state (not built per-request)
- Rate limiting: 60 queries/minute per IP (FR-64)
- Prometheus metrics: query totals, duration, cache hits (NFR)
"""

import time
import uuid
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, validator
from typing import Optional

from src.security.auth import TokenData, get_current_user
from src.security.rbac import has_permission
from src.audit.logger import audit_logger
from src.agent.orchestrator import AgentOrchestrator
from src.retrieval.cache import query_cache
from src.monitoring.metrics import record_query, record_cache_hit, record_cache_miss, record_agent_iteration

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_limiter():
    """Lazy import of limiter to avoid circular imports."""
    from src.api.main import limiter
    return limiter


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Query string (1–2000 chars)")
    session_id: Optional[str] = None
    conversation_history: Optional[list] = []
    top_k: Optional[int] = 10
    
    @validator('query')
    def query_not_empty_after_strip(cls, v):
        """Prevent empty or whitespace-only queries."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()
    
    @validator('conversation_history')
    def validate_history_size(cls, v):
        """Prevent excessively large conversation histories (>100KB or >50 turns)."""
        if v is None:
            return []
        if len(v) > 50:
            raise ValueError("Conversation history exceeds maximum (50 turns)")
        total_chars = sum(len(str(turn.get('content', ''))) for turn in v if isinstance(turn, dict))
        if total_chars > 100_000:  # 100KB max
            raise ValueError("Conversation history exceeds size limit (100KB)")
        return v[:50]  # Trim to last 50 turns
    
    @validator('top_k')
    def validate_top_k(cls, v):
        """Validate top_k bounds to prevent DoS."""
        if v is None:
            return 10
        if v < 1:
            raise ValueError("top_k must be at least 1")
        if v > 100:
            raise ValueError("top_k must not exceed 100")
        return v


class QueryResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    confidence: float
    citations: list
    sources: list
    reasoning_trace: list
    query_type: str
    duration_ms: float
    conflicting_info: list = []
    fallback_used: bool = False


def _get_orchestrator() -> AgentOrchestrator:
    """Retrieve the singleton AgentOrchestrator from shared app_state."""
    from src.api.main import app_state
    orchestrator = app_state.get("orchestrator")
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent orchestrator not initialized. Is the server still starting up?")
    return orchestrator


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    http_request: Request,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Submit a query to the KG-RAG agent.
    Returns the full answer with citations and reasoning trace.
    """
    if not has_permission(current_user.role, "query"):
        raise HTTPException(status_code=403, detail="Query permission denied")

    session_id = request.session_id or str(uuid.uuid4())
    start_time = time.time()

    # ── Cache lookup ────────────────────────────────────────
    cached_state = query_cache.get(request.query, current_user.user_id)
    if cached_state is not None:
        duration_ms = (time.time() - start_time) * 1000
        record_cache_hit()
        record_query(
            query_type="cached",
            user_role=current_user.role,
            status="success",
            duration_seconds=duration_ms / 1000,
        )
        return QueryResponse(
            session_id=session_id,
            query=request.query,
            answer=cached_state.get("final_answer", ""),
            confidence=cached_state.get("confidence", 0.0),
            citations=cached_state.get("citations", []),
            sources=cached_state.get("sources", []),
            reasoning_trace=cached_state.get("reasoning_trace", []),
            query_type=cached_state.get("query_type", "cached"),
            duration_ms=duration_ms,
            conflicting_info=cached_state.get("conflicting_info", []),
            fallback_used=cached_state.get("fallback_used", False),
        )
    record_cache_miss()

    # Audit log the query
    audit_logger.log_query(
        user_id=current_user.user_id,
        user_role=current_user.role,
        session_id=session_id,
        query=request.query,
    )

    orchestrator = _get_orchestrator()

    try:
        state = orchestrator.run(
            query=request.query,
            user_id=current_user.user_id,
            user_role=current_user.role,
            session_id=session_id,
            conversation_history=request.conversation_history or [],
        )
    except Exception as e:
        logger.error(f"Agent run failed: {e}")
        audit_logger.log_error(
            user_id=current_user.user_id,
            session_id=session_id,
            error_type="agent_failure",
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    duration_ms = (time.time() - start_time) * 1000

    # Audit log the answer
    audit_logger.log_answer(
        user_id=current_user.user_id,
        session_id=session_id,
        answer_length=len(state.get("final_answer", "")),
        confidence=state.get("confidence", 0.0),
        citation_count=len(state.get("citations", [])),
        has_conflicts=bool(state.get("conflicting_info")),
        duration_ms=duration_ms,
    )

    # Audit log agent reasoning
    audit_logger.log_agent_reasoning(
        user_id=current_user.user_id,
        session_id=session_id,
        reasoning_trace=state.get("reasoning_trace", []),
        tool_calls=state.get("tool_calls", []),
        iteration_count=state.get("iteration_count", 0),
    )

    # ── Prometheus metrics ───────────────────────────────────
    record_query(
        query_type=state.get("query_type", "unknown"),
        user_role=current_user.role,
        status="error" if state.get("error") else "success",
        duration_seconds=duration_ms / 1000,
    )
    if state.get("iteration_count", 0) > 1:
        for _ in range(state["iteration_count"] - 1):
            record_agent_iteration()

    # ── Cache result ────────────────────────────────────────
    if not state.get("fallback_used") and not state.get("error"):
        query_cache.set(request.query, current_user.user_id, state)

    return QueryResponse(
        session_id=session_id,
        query=request.query,
        answer=state.get("final_answer", ""),
        confidence=state.get("confidence", 0.0),
        citations=state.get("citations", []),
        sources=state.get("sources", []),
        reasoning_trace=state.get("reasoning_trace", []),
        query_type=state.get("query_type", ""),
        duration_ms=duration_ms,
        conflicting_info=state.get("conflicting_info", []),
        fallback_used=state.get("fallback_used", False),
    )



@router.websocket("/stream")
async def stream_endpoint(
    websocket: WebSocket,
    token: str,
):
    """
    WebSocket endpoint for streaming agent reasoning + answer.
    Client sends: {"query": "...", "session_id": "..."}
    Server streams: {"type": "reasoning_step"|"final_answer"|"error", "content": "..."}
    """
    await websocket.accept()

    # Validate token
    try:
        from src.security.auth import decode_token
        token_data = decode_token(token)
    except Exception:
        await websocket.send_json({"type": "error", "content": "Invalid token"})
        await websocket.close(code=1008)  # Policy Violation
        return

    # RBAC permission check — was missing, any valid JWT could query
    if not has_permission(token_data.role, "query"):
        await websocket.send_json({"type": "error", "content": "Query permission denied"})
        await websocket.close(code=1008)
        return

    session_id = str(uuid.uuid4())

    try:
        # Receive query
        raw = await websocket.receive_text()
        data = json.loads(raw)
        query = data.get("query", "")
        session_id = data.get("session_id", session_id)
        conversation_history = data.get("conversation_history", [])

        if not query:
            await websocket.send_json({"type": "error", "content": "Empty query"})
            return

        if len(query) > 2000:
            await websocket.send_json({"type": "error", "content": "Query exceeds maximum length of 2000 characters"})
            return

        audit_logger.log_query(
            user_id=token_data.user_id,
            user_role=token_data.role,
            session_id=session_id,
            query=query,
        )

        orchestrator = _get_orchestrator()

        # Stream agent events
        async for event in orchestrator.run_stream(
            query=query,
            user_id=token_data.user_id,
            user_role=token_data.role,
            session_id=session_id,
            conversation_history=conversation_history,
        ):
            await websocket.send_json(event)

        await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session={session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception as send_err:
            logger.warning(f"Failed to send WebSocket error response: {send_err}")
