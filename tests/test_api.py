"""
API Route Tests — health check, auth enforcement, RBAC, and admin endpoints.
"""

import os
import pytest


def test_health_check(test_client):
    """/health returns 200."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_query_requires_auth(test_client):
    """/api/query without token returns 403 or 401."""
    resp = test_client.post("/api/query", json={"query": "What is RBAC?"})
    assert resp.status_code in (401, 403)


def test_admin_stats_requires_admin(test_client, admin_headers):
    """/api/admin/stats is accessible with admin token."""
    resp = test_client.get("/api/admin/stats", headers=admin_headers)
    # May fail with 500 if Neo4j is not running, but must not fail with 401/403
    assert resp.status_code != 401
    assert resp.status_code != 403


def test_admin_stats_blocked_for_non_admin(test_client, set_test_env):
    """Non-admin users cannot access /api/admin/stats."""
    import importlib, src.security.auth as auth_mod
    importlib.reload(auth_mod)
    auth_mod.create_user("regularuser", "pass123", "user")

    resp = test_client.post(
        "/api/auth/token",
        data={"username": "regularuser", "password": "pass123"},
    )
    assert resp.status_code == 200
    user_token = resp.json()["access_token"]

    resp2 = test_client.get(
        "/api/admin/stats",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert resp2.status_code == 403


def test_create_user_via_admin(test_client, admin_headers):
    """/api/admin/users creates a new user successfully."""
    resp = test_client.post(
        "/api/admin/users",
        json={"username": "newuser_api_test", "password": "testpass", "role": "user"},
        headers=admin_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "created"
    assert body["username"] == "newuser_api_test"
    assert body["role"] == "user"


def test_create_duplicate_user_returns_409(test_client, admin_headers):
    """Creating a duplicate user returns 409 Conflict."""
    test_client.post(
        "/api/admin/users",
        json={"username": "dup_user", "password": "pass", "role": "user"},
        headers=admin_headers,
    )
    resp = test_client.post(
        "/api/admin/users",
        json={"username": "dup_user", "password": "pass", "role": "user"},
        headers=admin_headers,
    )
    assert resp.status_code == 409


def test_create_user_invalid_role_returns_400(test_client, admin_headers):
    """Invalid role returns 400."""
    resp = test_client.post(
        "/api/admin/users",
        json={"username": "bad_role_user", "password": "pass", "role": "superuser"},
        headers=admin_headers,
    )
    assert resp.status_code == 400


def test_audit_log_endpoint(test_client, admin_headers):
    """/api/admin/audit returns paginated log records."""
    resp = test_client.get("/api/admin/audit?limit=10", headers=admin_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "records" in body
    assert "count" in body


def test_ingest_requires_admin(test_client, set_test_env):
    """Non-admin cannot upload documents."""
    import importlib, src.security.auth as auth_mod
    importlib.reload(auth_mod)
    auth_mod.create_user("ingest_user", "pass", "user")

    resp = test_client.post(
        "/api/auth/token",
        data={"username": "ingest_user", "password": "pass"},
    )
    user_token = resp.json()["access_token"]

    resp2 = test_client.post(
        "/api/ingest",
        headers={"Authorization": f"Bearer {user_token}"},
        files={"file": ("test.txt", b"content", "text/plain")},
    )
    assert resp2.status_code == 403


def test_cors_no_wildcard(test_client):
    """CORS headers should not include a wildcard origin."""
    resp = test_client.options(
        "/health",
        headers={"Origin": "http://evil.com", "Access-Control-Request-Method": "GET"},
    )
    allow_origin = resp.headers.get("access-control-allow-origin", "")
    assert allow_origin != "*", "CORS wildcard must not be present in production mode"


def test_jwt_insecure_default_raises():
    """App startup raises RuntimeError if JWT secret is the default."""
    from src.api.main import _validate_configuration
    import os
    import pytest
    
    # Set the env var to the default insecure value
    os.environ["JWT_SECRET_KEY"] = "change-this-in-production-minimum-32-chars-required"
    
    with pytest.raises(RuntimeError, match="FATAL: JWT_SECRET_KEY"):
        _validate_configuration()


def test_query_too_long(test_client, set_test_env):
    """/api/query enforces max_length=2000."""
    import importlib, src.security.auth as auth_mod
    importlib.reload(auth_mod)
    auth_mod.create_user("queryuser", "pass123", "user")

    resp = test_client.post(
        "/api/auth/token",
        data={"username": "queryuser", "password": "pass123"},
    )
    user_token = resp.json()["access_token"]

    long_query = "A" * 2001
    resp2 = test_client.post(
        "/api/query",
        headers={"Authorization": f"Bearer {user_token}"},
        json={"query": long_query, "stream": False}
    )
    assert resp2.status_code == 422
    assert "String should have at most 2000 characters" in resp2.text
