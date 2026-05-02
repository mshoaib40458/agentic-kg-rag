"""
Auth — Phase 7 (Production)
JWT-based authentication middleware for FastAPI.
User management backed by a persistent JSON file store (data/users.json).
"""

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-in-production-minimum-32-chars-required")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

class _BcryptContext:
    def hash(self, password: str) -> str:
        # Validate bcrypt 72-byte limit before hashing
        encoded = password.encode("utf-8")
        if len(encoded) > 72:
            logger.warning(f"Password exceeds 72 bytes and will be truncated. Use shorter password.")
        return bcrypt.hashpw(encoded[:72], bcrypt.gensalt()).decode("utf-8")
        
    def verify(self, plain: str, hashed: str) -> bool:
        try:
            encoded = plain.encode("utf-8")
            # Apply same truncation as hash() for consistency
            return bcrypt.checkpw(encoded[:72], hashed.encode("utf-8"))
        except Exception:
            return False

pwd_context = _BcryptContext()
security = HTTPBearer()

# ── Persistent User Store ───────────────────────────────────────
USERS_FILE = Path(os.getenv("USERS_FILE_PATH", "data/users.json"))
_users_lock = threading.Lock()


def _load_users() -> dict[str, dict]:
    """Load users from the JSON file. Returns empty dict if file missing."""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: dict[str, dict]) -> None:
    """Persist the users dict to the JSON file (atomic write)."""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = USERS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)
    tmp.replace(USERS_FILE)


# ── Insecure default password constant ─────────────────────────
_DEFAULT_INSECURE_PASSWORD = "admin123"


def _ensure_admin() -> None:
    """Create the admin user with strong password enforcement."""
    import logging as _logging
    _log = _logging.getLogger(__name__)
    with _users_lock:
        users = _load_users()
        if not users:
            admin_username = os.getenv("ADMIN_USERNAME", "admin")
            admin_password = os.getenv("ADMIN_PASSWORD")
            
            # ENFORCE strong password requirement
            if not admin_password or admin_password == _DEFAULT_INSECURE_PASSWORD:
                raise RuntimeError(
                    "🚨 CRITICAL: ADMIN_PASSWORD environment variable is not set or uses default.\n"
                    "Set a strong password (minimum 16 characters) before starting:\n"
                    "export ADMIN_PASSWORD='<strong-password-min-16-chars>'\n"
                    "Refusing to start with weak credentials."
                )
            
            if len(admin_password) < 16:
                raise RuntimeError(f"ADMIN_PASSWORD too short ({len(admin_password)} chars). Use at least 16 characters.")
            
            # Check bcrypt 72-byte limit
            encoded = admin_password.encode("utf-8")
            if len(encoded) > 72:
                _log.warning(
                    f"Admin password exceeds 72 bytes when UTF-8 encoded. "
                    f"Only first 72 bytes will be used. Consider a shorter password to avoid truncation."
                )

            users[admin_username] = {
                "username": admin_username,
                "hashed_password": pwd_context.hash(admin_password),
                "role": "admin",
                "user_id": "usr_admin_001",
            }
            _save_users(users)
            _log.info("✓ Admin user created successfully")


def initialize_auth() -> None:
    """
    Explicitly initialize auth by creating the admin user if needed.
    Call this from FastAPI lifespan startup — NOT at module import time.
    """
    _ensure_admin()


# ── Models ─────────────────────────────────────────────────────

class TokenData(BaseModel):
    username: str
    user_id: str
    role: str


class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    user_id: str
    expires_in: int


# ── Core Auth Functions ─────────────────────────────────────────

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[dict]:
    with _users_lock:
        return _load_users().get(username)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role", "user")
        user_id: str = payload.get("user_id", "")
        if username is None:
            raise credentials_exception
        return TokenData(username=username, role=role, user_id=user_id)
    except JWTError:
        raise credentials_exception


# ── FastAPI Dependencies ────────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenData:
    """FastAPI dependency: validate JWT and return current user."""
    return decode_token(credentials.credentials)


async def require_admin(
    current_user: TokenData = Depends(get_current_user),
) -> TokenData:
    """FastAPI dependency: require admin role."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return current_user


def create_user(username: str, password: str, role: str = "user") -> dict:
    """Create a new user and persist to file (admin function)."""
    import uuid
    with _users_lock:
        users = _load_users()
        if username in users:
            raise ValueError(f"User '{username}' already exists")
        user = {
            "username": username,
            "hashed_password": get_password_hash(password),
            "role": role,
            "user_id": f"usr_{uuid.uuid4().hex[:8]}",
        }
        users[username] = user
        _save_users(users)
    return {"username": username, "role": role, "user_id": user["user_id"]}
