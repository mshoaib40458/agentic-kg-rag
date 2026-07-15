# Integration test configuration
import pytest
import os

# Set test environment variables before any imports
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-minimum-32-chars-long-for-testing")
os.environ.setdefault("ADMIN_PASSWORD", "StrongAdminPassword123!")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")


def pytest_configure(config):
    """Configure pytest for integration tests."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark integration tests."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)