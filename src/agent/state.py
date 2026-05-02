"""
Agent State — Phase 4
Typed AgentState schema for the LangGraph StateGraph.
This is the backbone of the entire agent loop.
"""

from typing import Any, Optional
from typing_extensions import TypedDict


class SourceRef(TypedDict):
    """A traceable reference to a source document or graph path."""
    chunk_id: str
    doc_id: str
    filename: str
    content_snippet: str
    source_type: str   # "vector" | "graph"
    score: float


class ToolCall(TypedDict):
    """Record of a single tool invocation."""
    tool_name: str
    input: dict
    output: Any
    success: bool
    error: Optional[str]
    duration_ms: float


class AgentState(TypedDict):
    """
    Complete typed state for the KG-RAG agent.
    Flows through all nodes in the LangGraph StateGraph.
    """
    # ── Input ──────────────────────────────────────────────────
    query: str
    user_id: str
    user_role: str                   # "admin" | "user" | "auditor"
    session_id: str
    conversation_history: list[dict] # Prior turns for follow-up support

    # ── Planning ───────────────────────────────────────────────
    query_type: str                  # factoid|multi-hop|comparative|policy|temporal|code
    plan: list[str]                  # CoT execution steps
    current_step: int

    # ── Execution ──────────────────────────────────────────────
    tool_calls: list[ToolCall]
    intermediate_results: list[dict]

    # ── Retrieval ──────────────────────────────────────────────
    vector_results: list[dict]
    graph_results: list[dict]
    reranked_results: list[dict]
    sources: list[SourceRef]

    # ── Reasoning ──────────────────────────────────────────────
    reasoning_trace: list[str]
    validation_passed: bool
    replan_needed: bool
    iteration_count: int

    # ── Output ─────────────────────────────────────────────────
    final_answer: str
    confidence: float
    citations: list[dict]
    conflicting_info: list[str]

    # ── Error Handling ─────────────────────────────────────────
    error: Optional[str]
    fallback_used: bool
