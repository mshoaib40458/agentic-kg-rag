"""
LangGraph Orchestrator — Phase 4
Wires all agent nodes into a LangGraph StateGraph.
Flow: START → Planner → Executor → Validator → (Replan? → Planner) → Synthesizer → END
"""

import logging
import uuid
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agent.executor import PlanExecutor
from src.agent.planner import QueryPlanner
from src.agent.state import AgentState
from src.agent.synthesizer import AnswerSynthesizer
from src.agent.tools import AgentTools
from src.agent.validator import ResultValidator

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3


def should_replan(state: AgentState) -> Literal["replan", "synthesize"]:
    """Edge condition: decide whether to replan or proceed to synthesis."""
    replan = state.get("replan_needed", False)
    valid = state.get("validation_passed", False)
    iters = state.get("iteration_count", 0)
    
    if replan and not valid and iters < MAX_ITERATIONS:
        return "replan"
    return "synthesize"


class AgentOrchestrator:
    """
    LangGraph StateGraph orchestrating the full KG-RAG agent loop.

    Flow:
        START
          ↓
        Planner (CoT query classification + plan generation)
          ↓
        Executor (tool dispatch: vector_search, graph_query, etc.)
          ↓
        Validator (result sufficiency check)
          ↓ [replan_needed?]
        Planner (re-plan) → Executor → Validator (max 3x)
          ↓ [validation_passed]
        Synthesizer (citation-aware answer)
          ↓
        END
    """

    def __init__(
        self,
        vector_store,
        graph_retriever,
        embedder,
        groq_client,
        llm_model: str = "llama-3.3-70b-versatile",
        ingest_model: str = "llama-3.1-8b-instant",
        hybrid_retriever=None,
        reranker=None,
    ):
        # Initialize components
        self.tools = AgentTools(
            vector_store=vector_store,
            graph_retriever=graph_retriever,
            embedder=embedder,
            groq_client=groq_client,
            llm_model=llm_model,
        )
        # Pass shared groq_client to planner and synthesizer (ARCH A4/A5 fix)
        self.planner = QueryPlanner(model=llm_model, groq_client=groq_client)
        self.executor = PlanExecutor(
            tools=self.tools,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
        )
        self.validator = ResultValidator()
        self.synthesizer = AnswerSynthesizer(model=llm_model, groq_client=groq_client)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph StateGraph."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("planner", self.planner.plan)
        workflow.add_node("executor", self.executor.execute)
        workflow.add_node("validator", self.validator.validate)
        workflow.add_node("synthesizer", self.synthesizer.synthesize)

        # Add edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "validator")

        # Conditional: replan or synthesize
        workflow.add_conditional_edges(
            "validator",
            should_replan,
            {
                "replan": "planner",
                "synthesize": "synthesizer",
            },
        )

        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    def run(
        self,
        query: str,
        user_id: str = "anonymous",
        user_role: str = "user",
        session_id: str = "",
        conversation_history: list = None,
    ) -> AgentState:
        """
        Run the full agent pipeline for a query.

        Args:
            query: User's natural language query.
            user_id: Authenticated user ID for audit logging.
            user_role: User's RBAC role.
            session_id: Session identifier for conversation tracking.
            conversation_history: Prior conversation turns.

        Returns:
            Final AgentState with answer, citations, reasoning trace.
        """
        initial_state: AgentState = {
            "query": query,
            "user_id": user_id,
            "user_role": user_role,
            "session_id": session_id or str(uuid.uuid4()),
            "conversation_history": conversation_history or [],
            "query_type": "",
            "plan": [],
            "current_step": 0,
            "tool_calls": [],
            "intermediate_results": [],
            "vector_results": [],
            "graph_results": [],
            "reranked_results": [],
            "sources": [],
            "reasoning_trace": [],
            "validation_passed": False,
            "replan_needed": False,
            "iteration_count": 0,
            "final_answer": "",
            "confidence": 0.0,
            "citations": [],
            "conflicting_info": [],
            "error": None,
            "fallback_used": False,
        }

        _preview = query[:80] + ("..." if len(query) > 80 else "")
        logger.info(f"Agent run started: query='{_preview}' user_role={user_role}")

        try:
            final_state = self.graph.invoke(initial_state)
            logger.info(
                f"Agent run complete: confidence={final_state.get('confidence', 0):.2f}, "
                f"citations={len(final_state.get('citations', []))}, "
                f"iterations={final_state.get('iteration_count', 0)}"
            )
            return final_state
        except Exception as e:
            logger.error(f"Agent orchestration failed: {e}")
            return {
                **initial_state,
                "final_answer": f"The agent encountered an error: {e}. Please retry.",
                "error": str(e),
                "fallback_used": True,
            }

    async def run_stream(
        self,
        query: str,
        user_id: str = "anonymous",
        user_role: str = "user",
        session_id: str = "",
        conversation_history: list = None,
    ):
        """
        Async generator for streaming agent reasoning steps and final answer.
        Yields dicts with type: 'reasoning_step' | 'final_answer' | 'error'
        """
        initial_state: AgentState = {
            "query": query,
            "user_id": user_id,
            "user_role": user_role,
            "session_id": session_id or str(uuid.uuid4()),
            "conversation_history": conversation_history or [],
            "query_type": "",
            "plan": [],
            "current_step": 0,
            "tool_calls": [],
            "intermediate_results": [],
            "vector_results": [],
            "graph_results": [],
            "reranked_results": [],
            "sources": [],
            "reasoning_trace": [],
            "validation_passed": False,
            "replan_needed": False,
            "iteration_count": 0,
            "final_answer": "",
            "confidence": 0.0,
            "citations": [],
            "conflicting_info": [],
            "error": None,
            "fallback_used": False,
        }

        try:
            # Run sync graph in thread pool to avoid blocking the async event loop
            import asyncio
            loop = asyncio.get_running_loop()
            final_state = await loop.run_in_executor(None, self.graph.invoke, initial_state)
            
            # Stream reasoning trace
            for step in final_state.get("reasoning_trace", []):
                yield {"type": "reasoning_step", "content": step}
            
            # Yield final answer
            yield {
                "type": "final_answer",
                "content": final_state.get("final_answer", ""),
                "confidence": final_state.get("confidence", 0.0),
                "citations": final_state.get("citations", []),
                "conflicting_info": final_state.get("conflicting_info", []),
                "sources": final_state.get("sources", []),
            }

        except Exception as e:
            logger.error(f"Stream failed: {e}")
            yield {"type": "error", "content": str(e)}
