"""
Shared pytest fixtures for the Agentic KG-RAG test suite.
Provides: app TestClient, mock orchestrator, temp VectorStore, mock Groq.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── Paths ────────────────────────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def set_test_env(tmp_path_factory):
    """Set environment variables for testing before any imports."""
    base = tmp_path_factory.mktemp("test_data")
    users_file = base / "users.json"
    faiss_index = base / "test.bin"
    faiss_meta = base / "test.json"
    os.environ["USERS_FILE_PATH"] = str(users_file)
    os.environ["FAISS_INDEX_PATH"] = str(faiss_index)
    os.environ["FAISS_METADATA_PATH"] = str(faiss_meta)
    os.environ["JWT_SECRET_KEY"] = "test-secret-minimum-32-chars-xxxx"
    os.environ["GROQ_API_KEY"] = "test-key"
    os.environ["REDIS_URL"] = "redis://localhost:9999"  # port that won't connect
    # Set a valid admin password so initialize_auth() doesn't raise
    os.environ["ADMIN_USERNAME"] = "admin"
    os.environ["ADMIN_PASSWORD"] = "TestAdminPass123!XY"  # 20 chars, satisfies >16 check

    # Bootstrap admin user now that env is set (was previously done at import-time)
    from src.security.auth import initialize_auth
    initialize_auth()

    yield
    # Cleanup handled by tmp_path_factory


@pytest.fixture(scope="session")
def test_client(set_test_env):
    import src.api.main  # Ensure module is loaded before patching its attributes
    with patch("src.api.main.VectorStore"), \
         patch("src.api.main.GraphRetriever"), \
         patch("src.api.main.DocumentEmbedder"), \
         patch("src.api.main.HybridRetriever"), \
         patch("src.api.main.CrossEncoderReranker"), \
         patch("groq.Groq"):
        from src.api.main import app
        with TestClient(app) as client:
            yield client


@pytest.fixture
def admin_token(test_client):
    """Get a valid admin JWT token."""
    resp = test_client.post(
        "/api/auth/token",
        data={"username": "admin", "password": os.getenv("ADMIN_PASSWORD", "TestAdminPass123!XY")},
    )
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return resp.json()["access_token"]


@pytest.fixture
def admin_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def temp_vector_store(tmp_path):
    """VectorStore backed by temp files."""
    from src.retrieval.vector_store import VectorStore
    return VectorStore(
        index_path=str(tmp_path / "test.bin"),
        metadata_path=str(tmp_path / "test.json"),
        embedding_dim=384,
    )


@pytest.fixture
def sample_chunks():
    """Minimal DocumentChunk-like objects for testing."""
    from dataclasses import dataclass, field

    @dataclass
    class FakeChunk:
        chunk_id: str
        doc_id: str
        filename: str
        content: str
        chunk_index: int = 0
        total_chunks: int = 1
        char_start: int = 0
        char_end: int = 100
        access_roles: list = field(default_factory=lambda: ["admin", "user"])
        embedding_model_id: str = "all-MiniLM-L6-v2"
        embedding_version: str = "v1"
        metadata: dict = field(default_factory=dict)

    return [
        FakeChunk("chunk-001", "doc-abc", "test.txt", "Alpha content about systems"),
        FakeChunk("chunk-002", "doc-abc", "test.txt", "Beta content about policies"),
        FakeChunk("chunk-003", "doc-xyz", "other.txt", "Gamma content about incidents"),
    ]


@pytest.fixture
def sample_embeddings():
    """Random normalized embeddings (dim=384)."""
    rng = np.random.default_rng(42)
    vecs = rng.random((3, 384)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms
