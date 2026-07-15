"""
Agent Executor — Phase 4
Executes the agent plan step-by-step using available tools.
Selects the right tool for each plan step based on query type and step content.
Supports max 3 re-planning cycles and graceful failure handling.
"""

import asyncio
import logging
import time
from src.agent.state import AgentState, ToolCall
from src.agent.tools import AgentTools
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

# Tool selection keywords
TOOL_SELECTION_MAP = {
    "graph_query":          ["graph", "relationship", "who owns", "depends", "caused", "violates", "path", "hop", "entity", "linked", "connected"],
    "code_search":          ["code", "function", "class", "api", "module", "implementation", "python", "method"],
    "entity_extraction":    ["extract entities", "identify entities", "named entities"],
    "source_verification":  ["verify", "confirm", "validate", "check claim", "is it true"],
    "vector_search":        [],  # Default fallback
}


class PlanExecutor:
    """
    Executes the step-by-step agent plan.
    Dynamically selects tools based on plan step content.
    Collects intermediate results and populates reasoning trace.
    After execution, merges via HybridRetriever and reranks via CrossEncoderReranker.
    """

    def __init__(
        self,
        tools: AgentTools,
        hybrid_retriever: "HybridRetriever" = None,
        reranker: "CrossEncoderReranker" = None,
    ):
        self.tools = tools
        # Accept injected singletons to avoid redundant construction per orchestrator init
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(vector_weight=0.6, graph_weight=0.4)
        self.reranker = reranker or CrossEncoderReranker(top_k=5)

    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute all plan steps and collect results.

        Args:
            state: Current AgentState with plan populated.

        Returns:
            Updated AgentState with tool_calls, intermediate_results, vector/graph results.
        """
        plan = state.get("plan", [])
        query = state["query"]
        query_type = state.get("query_type", "factoid")
        user_role = state.get("user_role", "user")
        reasoning_trace = state.get("reasoning_trace", [])
        tool_calls = state.get("tool_calls", [])
        intermediate_results = state.get("intermediate_results", [])
        vector_results = []
        graph_results = []

        max_tool_calls = 10
        call_count = 0

        # Use asyncio.to_thread for async-compatible execution (replaces ThreadPoolExecutor)
        # Run tool calls concurrently with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(5)

        async def run_tool(step_idx: int, step: str) -> tuple[str, dict, float]:
            nonlocal call_count
            if call_count >= max_tool_calls:
                return step, {"tool": "none", "success": False, "error": "Max tool calls reached"}, 0.0

            tool_name = self._select_tool(step, query_type)
            reasoning_trace.append(f"[EXECUTOR] Step {step_idx + 1}: {step}")
            reasoning_trace.append(f"[EXECUTOR] → Calling tool: {tool_name}")

            start = time.time()
            async with semaphore:
                result = await self._call_tool_async(tool_name, step, query, user_role, state)
            duration_ms = (time.time() - start) * 1000
            return tool_name, result, duration_ms

        # Create tasks for all steps
        tasks = [run_tool(idx, step) for idx, step in enumerate(plan) if call_count < max_tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for tool_name, result, duration_ms in results:
            if isinstance(result, Exception):
                logger.error(f"Tool {tool_name} raised exception: {result}")
                result = {"tool": tool_name, "success": False, "error": str(result)}
                duration_ms = 0.0

            call_count += 1

            tool_call: ToolCall = {
                "tool_name": tool_name,
                "input": {"step": step, "query": query},
                "output": result,
                "success": result.get("success", False),
                "error": result.get("error"),
                "duration_ms": duration_ms,
            }
            tool_calls.append(tool_call)
            intermediate_results.append(result)

            # Categorize results
            if tool_name == "vector_search" and result.get("success"):
                vector_results.extend(result.get("results", []))
            elif tool_name in ("graph_query",) and result.get("success"):
                graph_results.extend(result.get("paths", []))
                graph_results.extend(result.get("entities", []))

            reasoning_trace.append(
                f"[EXECUTOR] ✓ {tool_name} → "
                f"{result.get('result_count', result.get('entity_count', result.get('path_count', '?')))} results "
                f"in {duration_ms:.0f}ms"
            )

        logger.info(f"Execution complete: {call_count} tool calls")

        # ── Hybrid merge + rerank ──────────────────────────────
        try:
            merged = self.hybrid_retriever.merge(
                vector_results=vector_results,
                graph_results=graph_results,
                top_k=10,
            )
            reranked = self.reranker.rerank(
                query=query,
                candidates=merged,
                top_k=5,
            )
            # Convert back to dict list for state (keep compatibility)
            reranked_dicts = [
                {
                    "chunk_id": r.chunk_id,
                    "doc_id": r.doc_id,
                    "filename": r.filename,
                    "content": r.content,
                    "score": r.combined_score,
                    "rank": r.rank,
                    "metadata": r.metadata,
                    "source_type": r.source_type,
                }
                for r in reranked
            ]
            reasoning_trace.append(
                f"[EXECUTOR] 🎯 Reranked {len(merged)} merged results → top {len(reranked)} via cross-encoder"
            )
        except Exception as e:
            logger.warning(f"Hybrid merge/rerank failed (using raw results): {e}")
            reranked_dicts = vector_results[:5]

        return {
            **state,
            "tool_calls": tool_calls,
            "intermediate_results": intermediate_results,
            "vector_results": vector_results,
            "graph_results": graph_results,
            "reranked_results": reranked_dicts,
            "reasoning_trace": reasoning_trace,
            "current_step": len(plan),
        }

    def _select_tool(self, step: str, query_type: str) -> str:
        """Heuristic tool selection based on step text and query type."""
        step_lower = step.lower()

        # Check keyword mapping
        for tool, keywords in TOOL_SELECTION_MAP.items():
            if any(kw in step_lower for kw in keywords):
                return tool

        # Query type default routing
        if query_type in ("multi-hop", "comparative"):
            if "graph" not in step_lower:
                return "vector_search"
        elif query_type == "code":
            return "code_search"

        return "vector_search"  # Safe default

    def _extract_step_query(self, step: str, original_query: str) -> str:
        """
        Extract a targeted sub-query from a plan step text.
        Uses quoted strings or 'about X' patterns; falls back to original query.
        """
        import re
        # Try to find quoted phrase in step
        quoted = re.findall(r'["\u2018\u2019\u201c\u201d](.+?)["\u2018\u2019\u201c\u201d]', step)
        if quoted:
            return quoted[0]
        # Try 'for X', 'about X', 'on X', 'regarding X' patterns
        targets = re.findall(
            r'(?:for|about|on|regarding|related to|search|find|query)\s+(.+?)(?:\.|$)',
            step, re.IGNORECASE
        )
        if targets:
            candidate = targets[0].strip()
            # Only use if it's a meaningful sub-phrase (not too short or long)
            if 5 < len(candidate) < len(original_query) * 2:
                return candidate
        return original_query

    async def _call_tool_async(
        self,
        tool_name: str,
        step: str,
        query: str,
        user_role: str,
        state: AgentState,
    ) -> dict:
        """Async wrapper for tool dispatch using asyncio.to_thread."""
        targeted_query = self._extract_step_query(step, query)

        def _sync_call():
            return self._call_tool_sync(tool_name, targeted_query, user_role, state)

        try:
            return await asyncio.to_thread(_sync_call)
        except TypeError as e:
            # Type errors should fail fast, not be silent
            logger.error(f"Type error in tool {tool_name}: {e}")
            return {"tool": tool_name, "success": False, "error": f"Type error: {str(e)}"}
        except Exception as e:
            logger.error(f"Tool call {tool_name} raised exception: {e}")
            return {"tool": tool_name, "success": False, "error": str(e)}

    def _call_tool_sync(
        self,
        tool_name: str,
        query: str,
        user_role: str,
        state: AgentState,
    ) -> dict:
        """Synchronous tool dispatch with proper error handling and result validation."""
        try:
            if tool_name == "vector_search":
                result = self.tools.vector_search(query=query, top_k=10, user_role=user_role)
            elif tool_name == "graph_query":
                result = self.tools.graph_query(nl_query=query, user_role=user_role)
            elif tool_name == "entity_extraction":
                result = self.tools.entity_extraction(text=query)
            elif tool_name == "code_search":
                result = self.tools.code_search(query=query, user_role=user_role)
            elif tool_name == "source_verification":
                # Verify against accumulated vector results
                chunks = state.get("vector_results", [])[:5]
                result = self.tools.source_verification(claim=query, source_chunks=chunks)
            else:
                return {"tool": tool_name, "success": False, "error": f"Unknown tool: {tool_name}"}

            # Validate result shape before returning
            if not isinstance(result, dict) or "success" not in result:
                logger.error(f"Tool {tool_name} returned invalid shape: {type(result)}")
                return {"tool": tool_name, "success": False, "error": "Invalid tool response format"}

            return result

        except TypeError as e:
            # Type errors should fail fast, not be silent
            logger.error(f"Type error in tool {tool_name}: {e}")
            return {"tool": tool_name, "success": False, "error": f"Type error: {str(e)}"}
        except Exception as e:
            logger.error(f"Tool call {tool_name} raised exception: {e}")
            return {"tool": tool_name, "success": False, "error": str(e)}

