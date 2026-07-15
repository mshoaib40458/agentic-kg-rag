"""
Integration Tests — End-to-End API Tests
Tests the full request/response cycle including auth, ingestion, and query.
"""
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

# Set test environment before importing app
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-minimum-32-chars-long-for-testing")
os.environ.setdefault("ADMIN_PASSWORD", "StrongAdminPassword123!")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")


@pytest.fixture
def mock_groq():
    """Mock Groq API responses."""
    with patch("groq.Groq") as mock:
        client = MagicMock()
        mock.return_value = client
        client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"query_type": "factoid", "reasoning": "test", "plan": ["Step 1"], "primary_entities": [], "expected_tools": []}'))]
        ))
        yield client


@pytest.fixture
def mock_neo4j():
    """Mock Neo4j driver."""
    with patch("neo4j.GraphDatabase.driver") as mock:
        driver = MagicMock()
        mock.return_value = driver
        driver.verify_connectivity = MagicMock(return_value=True)
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=None)
        session.run = MagicMock(return_value=[])
        yield driver


@pytest.fixture
def mock_faiss():
    """Mock FAISS index operations."""
    with patch("faiss.IndexFlatIP") as mock_index, \
         patch("faiss.write_index"), \
         patch("faiss.read_index"):
        mock_idx = MagicMock()
        mock_idx.ntotal = 0
        mock_idx.is_trained = True
        mock_index.return_value = mock_idx
        yield mock_idx


@pytest.fixture
def test_client(mock_groq, mock_neo4j, mock_faiss):
    """Create test client with all external dependencies mocked."""
    from src.api.main import app
    return TestClient(app)


class TestAuthIntegration:
    """Test authentication flow."""

    def test_login_success(self, test_client):
        """Test successful login returns JWT token."""
        response = test_client.post(
            "/api/auth/token",
            data={"username": "admin", "password": "StrongAdminPassword123!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["role"] == "admin"

    def test_login_wrong_password(self, test_client):
        """Test login with wrong password returns 401."""
        response = test_client.post(
            "/api/auth/token",
            data={"username": "admin", "password": "wrongpassword"}
        )
        assert response.status_code == 401

    def test_login_nonexistent_user(self, test_client):
        """Test login with unknown user returns 401."""
        response = test_client.post(
            "/api/auth/token",
            data={"username": "nonexistent", "password": "anypassword"}
        )
        assert response.status_code == 401


class TestIngestionIntegration:
    """Test document ingestion flow."""

    def test_ingest_requires_auth(self, test_client):
        """Test ingestion endpoint requires authentication."""
        response = test_client.post("/api/ingest", files={"file": ("test.txt", b"content")})
        assert response.status_code == 403

    def test_ingest_invalid_file_type(self, test_client):
        """Test ingestion rejects unsupported file types."""
        # First login
        login = test_client.post("/api/auth/token", data={
            "username": "admin", "password": "StrongAdminPassword123!"
        })
        token = login.json()["access_token"]

        response = test_client.post(
            "/api/ingest",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("test.xyz", b"content")}
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]


class TestQueryIntegration:
    """Test query flow."""

    def test_query_requires_auth(self, test_client):
        """Test query endpoint requires authentication."""
        response = test_client.post("/api/query", json={"query": "test"})
        assert response.status_code == 403

    def test_query_validation(self, test_client):
        """Test query validation (max length, empty)."""
        login = test_client.post("/api/auth/token", data={
            "username": "admin", "password": "StrongAdminPassword123!"
        })
        token = login.json()["access_token"]

        # Empty query
        response = test_client.post(
            "/api/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": ""}
        )
        assert response.status_code == 422

        # Too long query
        long_query = "x" * 2001
        response = test_client.post(
            "/api/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": long_query}
        )
        assert response.status_code == 422


class TestAdminIntegration:
    """Test admin endpoints."""

    def test_admin_stats_requires_admin(self, test_client):
        """Test admin stats requires admin role."""
        # Login as regular user (if we had one)
        # For now test unauthenticated
        response = test_client.get("/api/admin/stats")
        assert response.status_code == 403

    def test_admin_stats_with_admin(self, test_client):
        """Test admin stats works with admin token."""
        login = test_client.post("/api/auth/token", data={
            "username": "admin", "password": "StrongAdminPassword123!"
        })
        token = login.json()["access_token"]

        response = test_client.get(
            "/api/admin/stats",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "vector_store" in data
        assert "graph_store" in data
        assert "cache_available" in data


class TestHealthCheck:
    """Test health check endpoint."""

    def test_health_check(self, test_client):
        """Test health check returns expected fields."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "vector_store" in data
        assert "cache_available" in data
        assert "graph_connected" in data


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint."""

    def test_metrics_endpoint(self, test_client):
        """Test /metrics returns Prometheus format."""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        assert "kgrag_" in response.text
        assert response.headers["content-type"].startswith("text/plain")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])