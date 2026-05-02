"""
Audit Logger — Phase 8
Structured JSON audit trail for all system events.
Logs: queries, retrieved chunks, graph paths, reasoning steps, prompts, LLM outputs.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog

# ── Setup structured logging ────────────────────────────────────
LOG_PATH = Path(os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _configure_structlog():
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


_configure_structlog()
_file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(message)s"))

_audit_logger = logging.getLogger("audit")
_audit_logger.addHandler(_file_handler)
_audit_logger.setLevel(logging.INFO)
_audit_logger.propagate = False


class AuditLogger:
    """
    Structured audit logger for compliance and traceability.
    Writes JSONL audit records to logs/audit.jsonl.
    Secrets (passwords, raw API keys) are NEVER logged.
    """

    def __init__(self, service_name: str = "kg-rag"):
        self.service = service_name

    def _write(self, event_type: str, data: dict):
        """Write a structured audit record."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": self.service,
            "event_type": event_type,
            **data,
        }
        _audit_logger.info(json.dumps(record, ensure_ascii=False, default=str))

    def log_query(
        self,
        user_id: str,
        user_role: str,
        session_id: str,
        query: str,
        query_type: Optional[str] = None,
    ):
        """Log an incoming user query."""
        self._write("user_query", {
            "user_id": user_id,
            "user_role": user_role,
            "session_id": session_id,
            "query": query[:500],  # Truncate for log size
            "query_type": query_type,
        })

    def log_retrieval(
        self,
        user_id: str,
        session_id: str,
        vector_result_count: int,
        graph_result_count: int,
        chunk_ids: list[str],
        doc_ids: list[str],
    ):
        """Log retrieval results for traceability."""
        self._write("retrieval", {
            "user_id": user_id,
            "session_id": session_id,
            "vector_results": vector_result_count,
            "graph_results": graph_result_count,
            "chunk_ids": chunk_ids[:20],  # Max 20 for log size
            "doc_ids": list(set(doc_ids))[:10],
        })

    def log_graph_traversal(
        self,
        user_id: str,
        session_id: str,
        cypher: str,
        paths: list[str],
        hop_depth: int,
    ):
        """Log graph traversal details."""
        self._write("graph_traversal", {
            "user_id": user_id,
            "session_id": session_id,
            "cypher": cypher[:500],
            "path_count": len(paths),
            "paths": paths[:10],
            "hop_depth": hop_depth,
        })

    def log_agent_reasoning(
        self,
        user_id: str,
        session_id: str,
        reasoning_trace: list[str],
        tool_calls: list[dict],
        iteration_count: int,
    ):
        """Log agent reasoning steps and tool selections."""
        self._write("agent_reasoning", {
            "user_id": user_id,
            "session_id": session_id,
            "iteration_count": iteration_count,
            "reasoning_steps": len(reasoning_trace),
            "reasoning_trace": reasoning_trace[-20:],  # Last 20 steps
            "tool_calls": [
                {"tool": t.get("tool_name"), "success": t.get("success"), "duration_ms": t.get("duration_ms")}
                for t in (tool_calls or [])[:10]
            ],
        })

    def log_llm_interaction(
        self,
        user_id: str,
        session_id: str,
        model: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ):
        """Log LLM prompt/response metadata (NOT content, for token tracking)."""
        self._write("llm_interaction", {
            "user_id": user_id,
            "session_id": session_id,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })

    def log_answer(
        self,
        user_id: str,
        session_id: str,
        answer_length: int,
        confidence: float,
        citation_count: int,
        has_conflicts: bool,
        duration_ms: float,
    ):
        """Log the final generated answer metadata."""
        self._write("answer_generated", {
            "user_id": user_id,
            "session_id": session_id,
            "answer_length": answer_length,
            "confidence": confidence,
            "citation_count": citation_count,
            "has_conflicts": has_conflicts,
            "duration_ms": duration_ms,
        })

    def log_ingestion(
        self,
        user_id: str,
        filename: str,
        doc_id: str,
        chunk_count: int,
        entity_count: int,
        relationship_count: int,
        duration_seconds: float,
    ):
        """Log document ingestion events."""
        self._write("document_ingestion", {
            "user_id": user_id,
            "filename": filename,
            "doc_id": doc_id,
            "chunk_count": chunk_count,
            "entity_count": entity_count,
            "relationship_count": relationship_count,
            "duration_seconds": duration_seconds,
        })

    def log_error(
        self,
        user_id: str,
        session_id: str,
        error_type: str,
        error_message: str,
        context: Optional[dict] = None,
    ):
        """Log system errors."""
        self._write("error", {
            "user_id": user_id,
            "session_id": session_id,
            "error_type": error_type,
            "error_message": error_message[:500],
            "context": context or {},
        })

    def read_logs(
        self,
        limit: int = 100,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Read audit logs with optional filtering (for admin API). Memory-efficient tail approach."""
        from collections import deque
        records = deque(maxlen=limit)
        if not LOG_PATH.exists():
            return []

        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if event_type and record.get("event_type") != event_type:
                            continue
                        if user_id and record.get("user_id") != user_id:
                            continue
                        records.append(record)
                    except json.JSONDecodeError:
                        continue
        except IOError as e:
            logger.warning(f"Failed to read audit logs: {e}")
            return []

        return list(reversed(list(records)))


# Singleton instance
audit_logger = AuditLogger()
