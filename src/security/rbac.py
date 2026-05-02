"""
RBAC — Phase 7
Role-Based Access Control for the KG-RAG system.
Defines roles, permissions, and document-level access checks.
"""

from enum import Enum
from typing import Optional


class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    AUDITOR = "auditor"


# Permission matrix: role → set of allowed actions
PERMISSIONS = {
    Role.ADMIN: {
        "query", "ingest", "delete", "admin", "audit",
        "view_reasoning", "manage_users", "system_config"
    },
    Role.USER: {
        "query", "view_reasoning"
    },
    Role.AUDITOR: {
        "query", "audit", "view_reasoning"
    },
}

# Document access levels — documents can be tagged with required_role
DOCUMENT_ACCESS_MATRIX = {
    "public": [Role.ADMIN, Role.USER, Role.AUDITOR],
    "internal": [Role.ADMIN, Role.USER, Role.AUDITOR],
    "restricted": [Role.ADMIN],
    "audit_only": [Role.ADMIN, Role.AUDITOR],
}


def has_permission(user_role: str, action: str) -> bool:
    """Check if a role has permission for an action."""
    try:
        role = Role(user_role)
        return action in PERMISSIONS.get(role, set())
    except ValueError:
        return False


def can_access_document(user_role: str, document_access_level: str = "internal") -> bool:
    """Check if a user role can access a document with the given access level."""
    try:
        role = Role(user_role)
        allowed_roles = DOCUMENT_ACCESS_MATRIX.get(document_access_level, [])
        return role in allowed_roles
    except ValueError:
        return False


def filter_results_by_role(results: list[dict], user_role: str) -> list[dict]:
    """
    Filter a list of retrieval results by user's RBAC role.
    Removes results from documents the user cannot access.
    """
    filtered = []
    for result in results:
        access_level = result.get("metadata", {}).get("access_level", "internal")
        if can_access_document(user_role, access_level):
            filtered.append(result)
    return filtered


def get_role_info(user_role: str) -> dict:
    """Return role metadata."""
    try:
        role = Role(user_role)
        return {
            "role": role.value,
            "permissions": list(PERMISSIONS.get(role, set())),
            "is_admin": role == Role.ADMIN,
        }
    except ValueError:
        return {"role": "unknown", "permissions": [], "is_admin": False}
