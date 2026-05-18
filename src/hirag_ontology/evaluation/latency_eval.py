"""Latency benchmark for retrieval, context building, and answer generation."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from hirag_ontology.evaluation.benchmark import load_benchmark
from hirag_ontology.evaluation.retrieval_eval import DEFAULT_GT_PATH, DEFAULT_KG_PATH
from hirag_ontology.llm import LLMClient
from hirag_ontology.pipeline.knowledge_graph import KnowledgeGraph
from hirag_ontology.pipeline.runner import demo_embedding_provider
from hirag_ontology.retrieval.answering import (
    answer_from_graph_context,
    build_graph_context,
    deterministic_answer_from_graph_context,
)
from hirag_ontology.retrieval.retriever import (
    EmbeddingProvider,
    HybridRetriever,
    RetrievalMode,
)


def measure_single_query(
    *,
    kg: KnowledgeGraph,
    query: str,
    top_k: int = 10,
    retrieval_mode: RetrievalMode = RetrievalMode.LEXICAL_STRUCTURAL,
    answer_mode: str = "deterministic",
    llm_client: LLMClient | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict[str, Any]:
    """Measure one end-to-end question-answering pass."""
    retriever = HybridRetriever(
        kg,
        embedding_provider or demo_embedding_provider(),
        mode=retrieval_mode,
    )

    retrieval_started = time.perf_counter()
    retrieved = retriever.retrieve(query, top_k=top_k)
    retrieval_s = time.perf_counter() - retrieval_started

    context_started = time.perf_counter()
    graph_context = build_graph_context(kg, retrieved, query=query)
    context_format_s = time.perf_counter() - context_started

    generation_started = time.perf_counter()
    if answer_mode == "deterministic":
        answer = deterministic_answer_from_graph_context(
            query=query,
            graph_context=graph_context,
            retrieved=retrieved,
        )
    elif answer_mode == "llm" and llm_client is not None:
        answer = answer_from_graph_context(
            llm_client,
            query=query,
            graph_context=graph_context,
        )
    else:
        msg = "answer_mode must be deterministic or llm with llm_client."
        raise ValueError(msg)
    generation_s = time.perf_counter() - generation_started

    return {
        "question": query,
        "retrieval_s": retrieval_s,
        "context_format_s": context_format_s,
        "generation_s": generation_s,
        "total_s": retrieval_s + context_format_s + generation_s,
        "n_entities_retrieved": len(retrieved),
        "answer_length_chars": len(answer),
    }


def run_latency_eval(
    *,
    kg_path: str | Path = DEFAULT_KG_PATH,
    gt_path: str | Path = DEFAULT_GT_PATH,
    n_queries: int = 20,
    top_k: int = 10,
    retrieval_mode: RetrievalMode = RetrievalMode.LEXICAL_STRUCTURAL,
    answer_mode: str = "deterministic",
    llm_client: LLMClient | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict[str, Any]:
    """Run latency benchmark over the first N benchmark questions."""
    kg = KnowledgeGraph.load(kg_path)
    questions = load_benchmark(gt_path)[:n_queries]
    per_query = [
        measure_single_query(
            kg=kg,
            query=question.question,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            answer_mode=answer_mode,
            llm_client=llm_client,
            embedding_provider=embedding_provider,
        )
        for question in questions
    ]
    return {
        "kg_path": str(kg_path),
        "gt_path": str(gt_path),
        "n_queries": len(per_query),
        "top_k": top_k,
        "retrieval_mode": retrieval_mode.value,
        "answer_mode": answer_mode,
        "aggregated": aggregate_timings(per_query),
        "per_query": per_query,
    }


def aggregate_timings(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate timing rows into mean/std/min/max tables."""
    keys = ["retrieval_s", "context_format_s", "generation_s", "total_s"]
    aggregated: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        aggregated[key] = {
            "mean": statistics.mean(values) if values else 0.0,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
    return aggregated


def save_latency_eval(results: dict[str, Any], out_dir: str | Path) -> None:
    """Save latency benchmark output."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latency_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_latency_summary(results: dict[str, Any]) -> None:
    """Print a compact latency table."""
    print("Latency metrics")
    print("stage                 mean    std     min     max")
    for stage, metrics in results["aggregated"].items():
        print(
            f"{stage:<20} "
            f"{metrics['mean']:.3f}  "
            f"{metrics['std']:.3f}  "
            f"{metrics['min']:.3f}  "
            f"{metrics['max']:.3f}"
        )


def main() -> None:
    """CLI entry point for latency evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run latency evaluation.")
    parser.add_argument("--kg", default=str(DEFAULT_KG_PATH))
    parser.add_argument("--gt", default=str(DEFAULT_GT_PATH))
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    results = run_latency_eval(
        kg_path=args.kg,
        gt_path=args.gt,
        n_queries=args.n,
        top_k=args.top_k,
    )
    print_latency_summary(results)
    save_latency_eval(results, args.out_dir)


if __name__ == "__main__":
    main()
