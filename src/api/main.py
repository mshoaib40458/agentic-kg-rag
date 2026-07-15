"""
FastAPI Main App — Phase 9 (Production)
Entry point with middleware, CORS, rate limiting, and OpenAPI docs.

Production Fixes Applied:
- JWT_SECRET_KEY guard: raises RuntimeError if default insecure key is used
- Singleton AgentOrchestrator: built once at startup, shared across all requests
- Richer /health endpoint: includes cache_available, graph_connected, groq_available, embedding_available, reranker_available status
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.api.routes import query, ingest, admin
from src.retrieval.vector_store import VectorStore
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.ingestion.embedder import DocumentEmbedder
from src.retrieval.cache import query_cache
from groq import Groq

logger = logging.getLogger(__name__)

# ── Rate Limiter ───────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── App State (shared instances) ───────────────────────────────
app_state = {}


def _validate_configuration() -> None:
    """
    Validate critical configuration at startup.
    Raises RuntimeError if required settings are insecure or missing.
    """
    secret = os.getenv("JWT_SECRET_KEY")
    if not secret:
        raise RuntimeError(
            "🚨 FATAL: JWT_SECRET_KEY environment variable is not set. "
            "Set a strong, unique secret in your .env file before starting the server. "
            "Minimum 32 characters recommended."
        )
    if len(secret) < 32:
        raise RuntimeError(
            f"🚨 FATAL: JWT_SECRET_KEY is too short ({len(secret)} chars). "
            "Use at least 32 characters for HS256 security."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup, cleanup on shutdown."""
    # ── Validate config before any resource initialization ──────
    _validate_configuration()

    # ── Initialize auth (deferred from import-time to avoid RuntimeError on missing env) ──
    from src.security.auth import initialize_auth
    initialize_auth()

    logger.info("🚀 KG-RAG Enterprise Assistant starting up...")

    # Initialize vector store
    vector_store = VectorStore(
        index_path=os.getenv("FAISS_INDEX_PATH", "data/faiss_index"),
        metadata_path=os.getenv("FAISS_METADATA_PATH", "data/faiss_metadata.json"),
        embedding_dim=int(os.getenv("EMBEDDING_DIMENSION", "384")),
    )
    try:
        vector_store.load()
        logger.info(f"✓ FAISS index loaded: {vector_store.get_stats()}")
    except FileNotFoundError:
        logger.info("FAISS index not found — will be created on first ingestion")

    # Initialize embedder (singleton — shared by both query and ingestion pipelines)
    embedder = DocumentEmbedder(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # Initialize graph retriever
    graph_retriever = GraphRetriever(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_username=os.getenv("NEO4J_USERNAME"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        llm_model=os.getenv("LLM_QUERY_MODEL", "llama-3.3-70b-versatile"),
    )
    if graph_retriever.verify_connectivity():
        logger.info("✓ Neo4j connection verified")
    else:
        logger.warning("⚠ Neo4j connection failed! Graph retrieval will be unavailable.")

    # Initialize Groq client (singleton — shared by planner, synthesizer, tools)
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # ── Retrieval singletons — built once, injected into executor ──
    hybrid_retriever = HybridRetriever(
        vector_weight=float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6")),
        graph_weight=float(os.getenv("HYBRID_GRAPH_WEIGHT", "0.4")),
    )
    reranker = CrossEncoderReranker(
        model_name=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        top_k=int(os.getenv("RERANK_TOP_K", "5")),
    )

    # ── Build singleton orchestrator ───────────────────────────
    from src.agent.orchestrator import AgentOrchestrator
    orchestrator = AgentOrchestrator(
        vector_store=vector_store,
        graph_retriever=graph_retriever,
        embedder=embedder,
        groq_client=groq_client,
        llm_model=os.getenv("LLM_QUERY_MODEL", "llama-3.3-70b-versatile"),
        ingest_model=os.getenv("LLM_INGEST_MODEL", "llama-3.1-8b-instant"),
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )

    # Store in app state
    app_state["vector_store"] = vector_store
    app_state["embedder"] = embedder
    app_state["graph_retriever"] = graph_retriever
    app_state["groq_client"] = groq_client
    app_state["orchestrator"] = orchestrator

    # Warm up the cache connection (non-blocking — cache.py handles failures gracefully)
    _ = await query_cache.check_available()

    # ── Health checks for external dependencies ──────────────────
    groq_available = False
    embedding_available = False
    reranker_available = False

    # Check Groq API
    try:
        groq_client.chat.completions.create(
            model=os.getenv("LLM_QUERY_MODEL", "llama-3.3-70b-versatile"),
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
            temperature=0.0,
        )
        groq_available = True
        logger.info("✓ Groq API connectivity verified")
    except Exception as e:
        logger.warning(f"⚠ Groq API check failed: {e}")

    # Check embedding model
    try:
        _ = embedder.get_dimension()
        embedding_available = True
        logger.info("✓ Embedding model loaded successfully")
    except Exception as e:
        logger.warning(f"⚠ Embedding model check failed: {e}")

    # Check reranker model
    try:
        reranker._load_model()
        if reranker._model is not None:
            reranker_available = True
            logger.info("✓ Cross-encoder reranker loaded successfully")
        else:
            logger.warning("⚠ Cross-encoder reranker not available (will use score fallback)")
    except Exception as e:
        logger.warning(f"⚠ Reranker model check failed: {e}")

    logger.info("✓ All services initialized")
    yield

    # Cleanup
    logger.info("KG-RAG shutting down...")
    graph_retriever.close()


# ── FastAPI App ─────────────────────────────────────────────────
app = FastAPI(
    title="Agentic KG-RAG Enterprise Assistant",
    description="Agentic Knowledge Graph RAG System — IEEE 830-1998 compliant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Attach rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "type": "about:blank",
            "title": "Internal Server Error",
            "status": 500,
            "detail": "An unexpected error occurred while processing the request.",
            "instance": request.url.path,
        },
    )

# ── CORS ────────────────────────────────────────────────────────
origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# ── Request Timing Middleware ────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


# ── Routes ─────────────────────────────────────────────────────
app.include_router(query.router, prefix="/api", tags=["Query"])
app.include_router(ingest.router, prefix="/api", tags=["Ingestion"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])


# ── Health Check ────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    """System health check endpoint."""
    vs = app_state.get("vector_store")
    gr = app_state.get("graph_retriever")
    embedder = app_state.get("embedder")
    groq_client = app_state.get("groq_client")
    stats = vs.get_stats() if vs else {}

    # Check graph connectivity (non-blocking)
    graph_connected = False
    if gr:
        try:
            driver = gr._get_driver()
            driver.verify_connectivity()
            graph_connected = True
        except Exception:
            graph_connected = False

    # Check Groq API connectivity
    groq_connected = False
    if groq_client:
        try:
            # Lightweight call to verify API key works
            groq_client.models.list()
            groq_connected = True
        except Exception:
            groq_connected = False

    # Check embedding model
    embedding_loaded = False
    if embedder:
        try:
            _ = embedder.get_dimension()
            embedding_loaded = True
        except Exception:
            embedding_loaded = False

    # Check reranker model
    reranker_loaded = False
    try:
        from sentence_transformers import CrossEncoder
        CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker_loaded = True
    except Exception:
        reranker_loaded = False

    return {
        "status": "healthy",
        "version": "1.0.0",
        "vector_store": stats,
        "cache_available": await query_cache.check_available(),
        "graph_connected": graph_connected,
        "groq_connected": groq_connected,
        "embedding_loaded": embedding_loaded,
        "reranker_loaded": reranker_loaded,
    }


# ── Prometheus Metrics Endpoint ───────────────────────────────────
@app.get("/metrics", include_in_schema=False, tags=["Monitoring"])
async def prometheus_metrics():
    """
    Prometheus scrape endpoint.
    Returns all kgrag_* metrics in Prometheus text format.
    No Docker required — configure Prometheus to scrape http://<host>:8000/metrics
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ── Auth endpoint ───────────────────────────────────────────────
from src.security.auth import authenticate_user, create_access_token
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends


@app.post("/api/auth/token", tags=["Auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate and receive JWT token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(data={
        "sub": user["username"],
        "role": user["role"],
        "user_id": user["user_id"],
    })
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "user_id": user["user_id"],
    }
