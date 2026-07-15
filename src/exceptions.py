"""
Common Exceptions — Shared exception hierarchy for the KG-RAG system.
Provides specific exception types to replace broad `except Exception` handlers.
"""

from typing import Optional


class KGException(Exception):
    """Base exception for all KG-RAG errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# ── Configuration Errors ──────────────────────────────────────────
class ConfigurationError(KGException):
    """Raised when required configuration is missing or invalid."""
    pass


class InsecureConfigurationError(ConfigurationError):
    """Raised when security configuration fails validation (weak secrets, etc.)."""
    pass


# ── Authentication/Authorization Errors ───────────────────────────
class AuthenticationError(KGException):
    """Raised when authentication fails."""
    pass


class AuthorizationError(KGException):
    """Raised when RBAC permission check fails."""
    pass


class TokenError(AuthenticationError):
    """Raised when JWT token is invalid, expired, or malformed."""
    pass


# ── External Service Errors ───────────────────────────────────────
class ExternalServiceError(KGException):
    """Base for external service failures (Groq, Neo4j, Redis, etc.)."""
    def __init__(self, service: str, message: str, details: Optional[dict] = None):
        super().__init__(f"{service}: {message}", details)
        self.service = service


class GroqAPIError(ExternalServiceError):
    """Raised when Groq API call fails."""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[dict] = None):
        super().__init__("Groq", message, details)
        self.status_code = status_code


class Neo4jError(ExternalServiceError):
    """Raised when Neo4j operation fails."""
    def __init__(self, message: str, query: Optional[str] = None, details: Optional[dict] = None):
        super().__init__("Neo4j", message, details)
        self.query = query


class RedisError(ExternalServiceError):
    """Raised when Redis operation fails."""
    pass


# ── Data/Storage Errors ───────────────────────────────────────────
class StorageError(KGException):
    """Base for storage-related failures (FAISS, file system)."""
    pass


class VectorStoreError(StorageError):
    """Raised when FAISS vector store operation fails."""
    pass


class IndexNotFoundError(VectorStoreError):
    """Raised when FAISS index file is missing."""
    pass


class DocumentProcessingError(KGException):
    """Raised when document parsing/chunking/embedding fails."""
    pass


class UnsupportedFormatError(DocumentProcessingError):
    """Raised when file format is not supported."""
    pass


# ── Agent/Orchestration Errors ────────────────────────────────────
class AgentError(KGException):
    """Base for agent orchestration failures."""
    pass


class PlanningError(AgentError):
    """Raised when query planning fails."""
    pass


class ExecutionError(AgentError):
    """Raised when tool execution fails."""
    pass


class ValidationError(AgentError):
    """Raised when result validation fails."""
    pass


class SynthesisError(AgentError):
    """Raised when answer synthesis fails."""
    pass


class MaxIterationsExceededError(AgentError):
    """Raised when agent exceeds max re-planning iterations."""
    pass


# ── Retrieval Errors ──────────────────────────────────────────────
class RetrievalError(KGException):
    """Base for retrieval failures."""
    pass


class GraphRetrievalError(RetrievalError):
    """Raised when graph query fails."""
    pass


class VectorRetrievalError(RetrievalError):
    """Raised when vector search fails."""
    pass


class CacheError(RetrievalError):
    """Raised when cache operation fails (non-fatal, logged)."""
    pass


# ── Ingestion Errors ──────────────────────────────────────────────
class IngestionError(KGException):
    """Raised when document ingestion pipeline fails."""
    pass


class EntityExtractionError(IngestionError):
    """Raised when NER fails."""
    pass


class RelationExtractionError(IngestionError):
    """Raised when relation extraction fails."""
    pass


# ── Utility: Exception Mapping ────────────────────────────────────
# Map external library exceptions to our hierarchy
EXTERNAL_EXCEPTION_MAP = {
    # Groq
    "groq.APIError": GroqAPIError,
    "groq.APIConnectionError": GroqAPIError,
    "groq.RateLimitError": GroqAPIError,
    "groq.AuthenticationError": GroqAPIError,
    "groq.BadRequestError": GroqAPIError,
    # Neo4j
    "neo4j.exceptions.Neo4jError": Neo4jError,
    "neo4j.exceptions.ServiceUnavailable": Neo4jError,
    "neo4j.exceptions.AuthError": Neo4jError,
    # Redis
    "redis.RedisError": RedisError,
    "redis.ConnectionError": RedisError,
    # FAISS/NumPy
    "faiss.Exception": VectorStoreError,
    # JSON
    "json.JSONDecodeError": KGException,
}


def map_exception(exc: Exception) -> KGException:
    """
    Map an external library exception to our exception hierarchy.
    Returns the original exception if no mapping exists.
    """
    exc_type = f"{exc.__class__.__module__}.{exc.__class__.__name__}"
    for ext_type, our_type in EXTERNAL_EXCEPTION_MAP.items():
        if ext_type in exc_type or isinstance(exc, eval(ext_type.split(".")[-1])):
            return our_type(str(exc))
    return exc