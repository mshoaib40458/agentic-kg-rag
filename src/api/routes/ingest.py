"""
Ingest Routes — Phase 9
POST /api/ingest — upload and ingest documents
"""

import os
import shutil
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.security.auth import TokenData, get_current_user
from src.security.rbac import has_permission
from src.audit.logger import audit_logger
from src.ingestion.pipeline import IngestionPipeline
from src.monitoring.metrics import record_ingestion

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = Path(os.getenv("DOCUMENTS_PATH", "data/documents"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


def _safe_filename(raw: str) -> str:
    """Strip directory components and sanitize filename to prevent path traversal."""
    import re
    # Take only the basename — kills any ../../ traversal
    name = Path(raw).name
    # Replace anything that isn't alphanumeric, dot, dash, or underscore
    name = re.sub(r"[^\w.\-]", "_", name)
    # Prevent hidden files or names that start with a dot
    name = name.lstrip(".")
    return name or "upload"


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    entities: int
    relationships: int
    duration_seconds: float
    status: str


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    access_level: str = "internal",
    current_user: TokenData = Depends(get_current_user),
):
    """
    Upload and ingest a document into the KG-RAG system.
    Requires 'ingest' permission (admin only by default).
    """
    if not has_permission(current_user.role, "ingest"):
        raise HTTPException(status_code=403, detail="Ingest permission denied (admin only)")

    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    # Validate file size
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB}MB limit")

    # Save file using a sanitized name — prevents path traversal
    doc_id = str(uuid.uuid4())
    safe_name = _safe_filename(file.filename)
    save_path = UPLOAD_DIR / f"{doc_id}_{safe_name}"
    # Final guard: ensure resolved path stays within UPLOAD_DIR
    if not str(save_path.resolve()).startswith(str(UPLOAD_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")
    with open(save_path, "wb") as f:
        f.write(contents)

    logger.info(f"File saved: {save_path} ({len(contents)/1024:.1f} KB)")

    try:
        from src.api.main import app_state
        pipeline = IngestionPipeline(
            vector_store=app_state.get("vector_store"),
            embedder=app_state.get("embedder"),   # Reuse singleton — avoids model reload per request
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            llm_model=os.getenv("LLM_INGEST_MODEL", "llama-3.1-8b-instant"),
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_username=os.getenv("NEO4J_USERNAME"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            run_kg_extraction=True,
        )

        result = await pipeline.ingest_file(
            file_path=str(save_path),
            metadata={
                "access_level": access_level,
                "uploaded_by": current_user.user_id,
                "doc_id": doc_id,
            },
            access_roles=["admin", "user", "auditor"],
        )

        audit_logger.log_ingestion(
            user_id=current_user.user_id,
            filename=file.filename,
            doc_id=doc_id,
            chunk_count=result.total_chunks,
            entity_count=result.total_entities,
            relationship_count=result.total_relationships,
            duration_seconds=result.duration_seconds,
        )
        record_ingestion(
            status="success" if not result.failed_documents else "partial_failure",
            duration_seconds=result.duration_seconds,
        )

        return IngestResponse(
            doc_id=doc_id,
            filename=file.filename,
            chunks=result.total_chunks,
            entities=result.total_entities,
            relationships=result.total_relationships,
            duration_seconds=result.duration_seconds,
            status="success" if not result.failed_documents else "partial_failure",
        )

    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        record_ingestion(status="failure", duration_seconds=0.0)
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
