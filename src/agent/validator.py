"""
Agent Validator — Phase 4
Validates intermediate results after execution.
Triggers re-planning if results are insufficient.
Max 3 re-plan cycles enforced.
"""

import logging
from src.agent.state import AgentState

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3
MIN_VECTOR_RESULTS = 1
MIN_GRAPH_RESULTS = 0  # Graph can be empty for factoid queries


class ResultValidator:
    """
    Validates the sufficiency of retrieved results.
    Decides whether to proceed to synthesis or trigger re-planning.
    """

    def validate(self, state: AgentState) -> AgentState:
        """
        Validate current execution results.

        Args:
            state: Current AgentState post-execution.

        Returns:
            Updated state with validation_passed and replan_needed flags.
        """
        reasoning_trace = state.get("reasoning_trace", [])
        vector_results = state.get("vector_results", [])
        graph_results = state.get("graph_results", [])
        iteration_count = state.get("iteration_count", 1)
        query_type = state.get("query_type", "factoid")

        issues = []

        # Check iteration limit
        if iteration_count >= MAX_ITERATIONS:
            reasoning_trace.append(
                f"[VALIDATOR] Max iterations ({MAX_ITERATIONS}) reached — forcing synthesis"
            )
            return {
                **state,
                "validation_passed": True,
                "replan_needed": False,
                "reasoning_trace": reasoning_trace,
                "fallback_used": len(vector_results) == 0,
            }

        # Check vector results
        if len(vector_results) < MIN_VECTOR_RESULTS:
            issues.append("Insufficient vector search results")

        # For multi-hop queries, graph paths are expected (not just entities)
        if query_type in ("multi-hop", "comparative") and iteration_count == 1:
            # Count only path-type results (dicts with path_string key)
            path_results = [r for r in graph_results if isinstance(r, dict) and "path_string" in r]
            if len(path_results) == 0:
                issues.append("No graph paths found for multi-hop/comparative query")

        # Check for too many tool failures
        tool_calls = state.get("tool_calls", [])
        failed_tools = [t for t in tool_calls if not t.get("success", True)]
        if len(failed_tools) > len(tool_calls) / 2 and len(tool_calls) > 0:
            issues.append(f"{len(failed_tools)}/{len(tool_calls)} tool calls failed")

        if issues:
            reasoning_trace.append(f"[VALIDATOR] ✗ Validation failed: {'; '.join(issues)}")
            reasoning_trace.append(f"[VALIDATOR] → Triggering re-planning (iteration {iteration_count + 1})")
            logger.warning(f"Validation failed: {issues} — triggering replan")
            return {
                **state,
                "validation_passed": False,
                "replan_needed": True,
                "reasoning_trace": reasoning_trace,
            }

        reasoning_trace.append(
            f"[VALIDATOR] ✓ Validation passed: "
            f"{len(vector_results)} vector results, {len(graph_results)} graph results"
        )
        logger.info(f"Validation passed: {len(vector_results)} vec, {len(graph_results)} graph results")

        return {
            **state,
            "validation_passed": True,
            "replan_needed": False,
            "reasoning_trace": reasoning_trace,
        }
