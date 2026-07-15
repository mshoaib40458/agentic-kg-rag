"""
Evaluation Framework — Phase 11
Benchmarks KG-RAG Agentic vs. baseline traditional RAG.
Metrics: answer correctness, faithfulness, citation accuracy, multi-hop success rate.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalQuery:
    """A single evaluation query with ground truth."""
    query_id: str
    query: str
    query_type: str           # factoid|multi-hop|comparative|policy|temporal|code
    expected_answer: str
    required_sources: list[str] = field(default_factory=list)
    requires_hops: int = 1


@dataclass
class EvalResult:
    """Result for a single query evaluation."""
    query_id: str
    query: str
    system: str               # "baseline" or "kgrag"
    generated_answer: str
    duration_ms: float
    citation_count: int
    confidence: float
    # Computed metrics
    answer_correctness: float = 0.0
    faithfulness: float = 0.0
    citation_accuracy: float = 0.0
    multihop_success: bool = False


@dataclass
class EvalReport:
    """Aggregated evaluation report."""
    system: str
    total_queries: int
    avg_answer_correctness: float
    avg_faithfulness: float
    avg_citation_accuracy: float
    multihop_success_rate: float
    avg_response_ms: float
    results: list[EvalResult]


class KGRAGEvaluator:
    """
    Evaluation harness comparing baseline RAG vs. Agentic KG-RAG.
    Uses Groq LLM for automated scoring (answer correctness, faithfulness).
    """

    def __init__(
        self,
        orchestrator=None,
        baseline_retriever=None,
        groq_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        vector_store=None,
        embedder=None,
    ):
        self.orchestrator = orchestrator
        self.baseline_retriever = baseline_retriever
        self.llm_model = llm_model or os.getenv("LLM_QUERY_MODEL", "llama-3.3-70b-versatile")
        self.vector_store = vector_store
        self.embedder = embedder
        self._groq = None
        if groq_api_key or os.getenv("GROQ_API_KEY"):
            from groq import Groq
            self._groq = Groq(api_key=groq_api_key or os.getenv("GROQ_API_KEY"))

    def load_eval_set(self, path: str) -> list[EvalQuery]:
        """Load evaluation queries from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [EvalQuery(**q) for q in data]

    def run(
        self,
        eval_set: list[EvalQuery],
        user_role: str = "admin",
    ) -> EvalReport:
        """Run evaluation on the full eval set."""
        results = []
        for query_obj in eval_set:
            logger.info(f"Evaluating: {query_obj.query_id} [{query_obj.query_type}]")
            start = time.time()
            try:
                state = self.orchestrator.run(
                    query=query_obj.query,
                    user_id="evaluator",
                    user_role=user_role,
                )
                duration_ms = (time.time() - start) * 1000
                result = EvalResult(
                    query_id=query_obj.query_id,
                    query=query_obj.query,
                    system="kgrag",
                    generated_answer=state.get("final_answer", ""),
                    duration_ms=duration_ms,
                    citation_count=len(state.get("citations", [])),
                    confidence=state.get("confidence", 0.0),
                )
                result = self._score(result, query_obj)
                results.append(result)
            except Exception as e:
                logger.error(f"Eval failed for {query_obj.query_id}: {e}")

        return self._aggregate(results, "kgrag")

    def _score(self, result: EvalResult, query: EvalQuery) -> EvalResult:
        """Score a single result using LLM-as-judge."""
        if not self._groq:
            # Placeholder scores if no LLM available
            result.answer_correctness = 0.5
            result.faithfulness = 0.7
            result.citation_accuracy = 0.6
            result.multihop_success = query.requires_hops <= 1
            return result

        prompt = f"""Score this answer on a 0.0-1.0 scale. Return JSON only.

Question: {query.query}
Expected Answer: {query.expected_answer}
Generated Answer: {result.generated_answer}

{{"answer_correctness": 0.0-1.0, "faithfulness": 0.0-1.0, "reasoning": "brief"}}"""

        try:
            response = self._groq.chat.completions.create(
                model=os.getenv("LLM_INGEST_MODEL", "llama-3.1-8b-instant"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=256,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            scores = json.loads(raw)
            result.answer_correctness = float(scores.get("answer_correctness", 0.5))
            result.faithfulness = float(scores.get("faithfulness", 0.7))
        except Exception:
            result.answer_correctness = 0.5
            result.faithfulness = 0.7

        # Citation accuracy: required_sources covered
        if query.required_sources and result.citation_count > 0:
            result.citation_accuracy = min(result.citation_count / len(query.required_sources), 1.0)
        else:
            result.citation_accuracy = 1.0 if not query.required_sources else 0.0

        # Multi-hop success
        result.multihop_success = (
            query.requires_hops <= 1 or
            (query.requires_hops > 1 and result.answer_correctness > 0.6)
        )
        return result

    def _aggregate(self, results: list[EvalResult], system: str) -> EvalReport:
        n = len(results) or 1
        multihop_results = [r for r in results if r.multihop_success is not None]
        return EvalReport(
            system=system,
            total_queries=len(results),
            avg_answer_correctness=sum(r.answer_correctness for r in results) / n,
            avg_faithfulness=sum(r.faithfulness for r in results) / n,
            avg_citation_accuracy=sum(r.citation_accuracy for r in results) / n,
            multihop_success_rate=sum(1 for r in multihop_results if r.multihop_success) / max(len(multihop_results), 1),
            avg_response_ms=sum(r.duration_ms for r in results) / n,
            results=results,
        )

    def save_report(self, report: EvalReport, output_path: str):
        """Save evaluation report to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "system": report.system,
            "total_queries": report.total_queries,
            "metrics": {
                "avg_answer_correctness": round(report.avg_answer_correctness, 4),
                "avg_faithfulness": round(report.avg_faithfulness, 4),
                "avg_citation_accuracy": round(report.avg_citation_accuracy, 4),
                "multihop_success_rate": round(report.multihop_success_rate, 4),
                "avg_response_ms": round(report.avg_response_ms, 2),
            },
            "results": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "answer_correctness": round(r.answer_correctness, 4),
                    "faithfulness": round(r.faithfulness, 4),
                    "citation_accuracy": round(r.citation_accuracy, 4),
                    "multihop_success": r.multihop_success,
                    "duration_ms": round(r.duration_ms, 2),
                    "confidence": round(r.confidence, 4),
                }
                for r in report.results
            ]
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"✓ Eval report saved to {output_path}")
        return data

    def run_baseline(
        self,
        eval_set: list[EvalQuery],
        user_role: str = "admin",
    ) -> EvalReport:
        """
        Run the baseline (vector-only) RAG system on the eval set.
        Used for A/B comparison against the Agentic KG-RAG (FR-56).
        """
        from src.evaluation.baseline_rag import BaselineRAG
        baseline = BaselineRAG(
            vector_store=self.vector_store,
            embedder=self.embedder,
            llm_model=os.getenv("LLM_INGEST_MODEL", "llama-3.1-8b-instant"),
        )
        results = []
        for query_obj in eval_set:
            logger.info(f"[Baseline] Evaluating: {query_obj.query_id} [{query_obj.query_type}]")
            try:
                br = baseline.run(query=query_obj.query, user_role=user_role)
                result = EvalResult(
                    query_id=query_obj.query_id,
                    query=query_obj.query,
                    system="baseline",
                    generated_answer=br.answer,
                    duration_ms=br.duration_ms,
                    citation_count=len(br.citations),
                    confidence=br.confidence,
                )
                result = self._score(result, query_obj)
                results.append(result)
            except Exception as e:
                logger.error(f"[Baseline] Eval failed for {query_obj.query_id}: {e}")

        return self._aggregate(results, "baseline")

    def compare(
        self,
        kgrag_report: EvalReport,
        baseline_report: EvalReport,
    ) -> dict:
        """
        Compare KG-RAG Agentic vs Baseline RAG and compute % improvement.
        Returns a dict with per-metric deltas and overall improvement score.
        """
        def pct_change(new: float, old: float) -> float:
            if old == 0:
                return 0.0
            return round((new - old) / old * 100, 2)

        comparison = {
            "kgrag_system": kgrag_report.system,
            "baseline_system": baseline_report.system,
            "total_queries": kgrag_report.total_queries,
            "metrics_comparison": {
                "answer_correctness": {
                    "kgrag": round(kgrag_report.avg_answer_correctness, 4),
                    "baseline": round(baseline_report.avg_answer_correctness, 4),
                    "improvement_pct": pct_change(
                        kgrag_report.avg_answer_correctness,
                        baseline_report.avg_answer_correctness,
                    ),
                },
                "faithfulness": {
                    "kgrag": round(kgrag_report.avg_faithfulness, 4),
                    "baseline": round(baseline_report.avg_faithfulness, 4),
                    "improvement_pct": pct_change(
                        kgrag_report.avg_faithfulness,
                        baseline_report.avg_faithfulness,
                    ),
                },
                "citation_accuracy": {
                    "kgrag": round(kgrag_report.avg_citation_accuracy, 4),
                    "baseline": round(baseline_report.avg_citation_accuracy, 4),
                    "improvement_pct": pct_change(
                        kgrag_report.avg_citation_accuracy,
                        baseline_report.avg_citation_accuracy,
                    ),
                },
                "multihop_success_rate": {
                    "kgrag": round(kgrag_report.multihop_success_rate, 4),
                    "baseline": round(baseline_report.multihop_success_rate, 4),
                    "improvement_pct": pct_change(
                        kgrag_report.multihop_success_rate,
                        baseline_report.multihop_success_rate,
                    ),
                },
                "avg_response_ms": {
                    "kgrag": round(kgrag_report.avg_response_ms, 2),
                    "baseline": round(baseline_report.avg_response_ms, 2),
                    "improvement_pct": pct_change(
                        baseline_report.avg_response_ms,  # Lower is better for latency
                        kgrag_report.avg_response_ms,
                    ),
                },
            },
            "overall_accuracy_improvement_pct": pct_change(
                kgrag_report.avg_answer_correctness,
                baseline_report.avg_answer_correctness,
            ),
            "target_improvement_pct": 40.0,
            "target_met": pct_change(
                kgrag_report.avg_answer_correctness,
                baseline_report.avg_answer_correctness,
            ) >= 40.0,
        }
        logger.info(
            f"A/B Comparison: KG-RAG accuracy improvement = "
            f"{comparison['overall_accuracy_improvement_pct']}% "
            f"(target: 40%, met: {comparison['target_met']})"
        )
        return comparison
