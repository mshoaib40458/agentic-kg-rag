"""
Auth Tests — test JWT generation, verification, roles, and persistent user store.
"""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch


def test_authenticate_valid_admin(set_test_env):
    """Admin credentials from env should authenticate successfully."""
    from src.security.auth import authenticate_user
    user = authenticate_user("admin", os.getenv("ADMIN_PASSWORD", "admin123"))
    assert user is not None
    assert user["role"] == "admin"
    assert user["username"] == "admin"


def test_authenticate_wrong_password(set_test_env):
    """Wrong password returns None, not an exception."""
    from src.security.auth import authenticate_user
    result = authenticate_user("admin", "totally-wrong-password")
    assert result is None


def test_authenticate_unknown_user(set_test_env):
    """Unknown username returns None."""
    from src.security.auth import authenticate_user
    result = authenticate_user("nobody", "password")
    assert result is None


def test_create_and_retrieve_user(set_test_env, tmp_path):
    """create_user() persists to file; get_user() retrieves it."""
    # Point to a fresh temp file for this test
    fresh_path = tmp_path / "users_test.json"
    with patch.dict(os.environ, {"USERS_FILE_PATH": str(fresh_path)}):
        # Need to reimport with new env
        import importlib, src.security.auth as auth_mod
        importlib.reload(auth_mod)

        auth_mod.create_user("alice", "password123", "user")
        user = auth_mod.get_user("alice")
        assert user is not None
        assert user["role"] == "user"
        assert user["username"] == "alice"

        # Verify it's actually saved to file
        assert fresh_path.exists()
        data = json.loads(fresh_path.read_text())
        assert "alice" in data


def test_create_duplicate_user_raises(set_test_env):
    """Creating a user with a duplicate username raises ValueError."""
    import importlib, src.security.auth as auth_mod
    importlib.reload(auth_mod)
    import pytest
    # Admin already exists
    with pytest.raises(ValueError, match="already exists"):
        auth_mod.create_user("admin", "newpass", "user")


def test_token_creation_and_decode(set_test_env):
    """create_access_token() produces a decodable JWT with correct claims."""
    from src.security.auth import create_access_token, decode_token
    token = create_access_token({"sub": "testuser", "role": "user", "user_id": "usr_001"})
    assert isinstance(token, str)

    token_data = decode_token(token)
    assert token_data.username == "testuser"
    assert token_data.role == "user"
    assert token_data.user_id == "usr_001"


def test_invalid_token_raises(set_test_env):
    """A tampered token raises HTTPException 401."""
    from src.security.auth import decode_token
    from fastapi import HTTPException
    import pytest
    with pytest.raises(HTTPException) as exc_info:
        decode_token("invalid.jwt.token")
    assert exc_info.value.status_code == 401


def test_login_endpoint_returns_200(test_client):
    """POST /api/auth/token with valid credentials returns 200 and a token."""
    resp = test_client.post(
        "/api/auth/token",
        data={"username": "admin", "password": os.getenv("ADMIN_PASSWORD", "admin123")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["role"] == "admin"


def test_login_endpoint_wrong_password_returns_401(test_client):
    """POST /api/auth/token with wrong password returns 401, not 500."""
    resp = test_client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "wrongpass"},
    )
    assert resp.status_code == 401
    assert "detail" in resp.json()
