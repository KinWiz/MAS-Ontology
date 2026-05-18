"""RAG answer-generation evaluation with deterministic fallback metrics."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from hirag_ontology.evaluation.benchmark import load_benchmark
from hirag_ontology.evaluation.llm_judge import (
    judge_context_precision,
    judge_faithfulness,
)
from hirag_ontology.evaluation.retrieval_eval import (
    DEFAULT_GT_PATH,
    DEFAULT_KG_PATH,
    entity_label_set,
    normalize_eval_label,
)
from hirag_ontology.llm import LLMClient
from hirag_ontology.pipeline.knowledge_graph import KnowledgeGraph
from hirag_ontology.pipeline.runner import demo_embedding_provider
from hirag_ontology.retrieval.answering import (
    answer_from_graph_context,
    build_graph_context,
    deterministic_answer_from_graph_context,
)
from hirag_ontology.retrieval.retriever import HybridRetriever, RetrievalMode


def run_generation_eval(
    *,
    kg_path: str | Path = DEFAULT_KG_PATH,
    gt_path: str | Path = DEFAULT_GT_PATH,
    top_k: int = 10,
    retrieval_mode: RetrievalMode = RetrievalMode.LEXICAL_STRUCTURAL,
    answer_mode: str = "deterministic",
    llm_client: LLMClient | None = None,
    judge_client: LLMClient | None = None,
    n_questions: int | None = None,
) -> dict[str, Any]:
    """Evaluate generated answers on faithfulness, relevance, and context quality."""
    started = time.perf_counter()
    kg = KnowledgeGraph.load(kg_path)
    questions = load_benchmark(gt_path)
    if n_questions is not None:
        questions = questions[:n_questions]

    retriever = HybridRetriever(
        kg,
        demo_embedding_provider(),
        mode=retrieval_mode,
    )
    per_question: list[dict[str, Any]] = []
    accumulators: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevance": [],
        "context_precision": [],
        "context_recall": [],
    }

    for question in questions:
        retrieved = retriever.retrieve(question.question, top_k=top_k)
        graph_context = build_graph_context(kg, retrieved, query=question.question)
        if answer_mode == "deterministic":
            answer = deterministic_answer_from_graph_context(
                query=question.question,
                graph_context=graph_context,
                retrieved=retrieved,
            )
        elif answer_mode == "llm" and llm_client is not None:
            answer = answer_from_graph_context(
                llm_client,
                query=question.question,
                graph_context=graph_context,
            )
        else:
            msg = "answer_mode must be deterministic or llm with llm_client."
            raise ValueError(msg)

        if judge_client is None:
            faithfulness = deterministic_faithfulness(graph_context, answer)
            context_precision = exact_context_precision(
                kg,
                [result.entity_id for result in retrieved],
                question.relevant_entity_labels,
            )
        else:
            faithfulness = judge_faithfulness(
                judge_client,
                question=question.question,
                context=graph_context,
                answer=answer,
            ).score
            context_precision = judge_context_precision(
                judge_client,
                question=question.question,
                retrieved_entities=[result.entity.label for result in retrieved],
            ).score

        answer_relevance = lexical_relevance(question.question, answer)
        context_recall = exact_context_recall(
            kg,
            [result.entity_id for result in retrieved],
            question.relevant_entity_labels,
        )

        metric_values = {
            "faithfulness": round(faithfulness, 4),
            "answer_relevance": round(answer_relevance, 4),
            "context_precision": round(context_precision, 4),
            "context_recall": round(context_recall, 4),
        }
        row = {
            "id": question.id,
            "question": question.question,
            "type": question.question_type,
            **metric_values,
            "answer_preview": answer[:500],
            "retrieved_entities": [result.entity.label for result in retrieved],
        }
        per_question.append(row)
        for metric, value in metric_values.items():
            accumulators[metric].append(value)

    return {
        "kg_path": str(kg_path),
        "gt_path": str(gt_path),
        "top_k": top_k,
        "retrieval_mode": retrieval_mode.value,
        "answer_mode": answer_mode,
        "judge": "llm" if judge_client is not None else "deterministic",
        "elapsed_s": time.perf_counter() - started,
        "summary": {
            metric: _mean(values)
            for metric, values in accumulators.items()
        },
        "per_question": per_question,
        "n_questions": len(questions),
    }


def deterministic_faithfulness(context: str, answer: str) -> float:
    """Estimate support as the fraction of answer sentences grounded in context."""
    if not answer.strip():
        return 0.0
    if "not supported by the graph context" in answer.casefold():
        return 1.0
    context_tokens = content_tokens(context)
    if not context_tokens:
        return 0.0
    claims = [
        sentence
        for sentence in re.split(r"[.!?;\n]+", answer)
        if content_tokens(sentence)
    ]
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        claim_tokens = content_tokens(claim)
        overlap = claim_tokens & context_tokens
        if len(overlap) / max(len(claim_tokens), 1) >= 0.35:
            supported += 1
    return supported / len(claims)


def lexical_relevance(question: str, answer: str) -> float:
    """Return token-overlap F1 between question and answer."""
    question_tokens = content_tokens(question)
    answer_tokens = content_tokens(answer)
    if not question_tokens or not answer_tokens:
        return 0.0
    overlap = question_tokens & answer_tokens
    precision = len(overlap) / len(answer_tokens)
    recall = len(overlap) / len(question_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def exact_context_precision(
    kg: KnowledgeGraph,
    retrieved_ids: list[str],
    relevant_labels: list[str],
) -> float:
    """Return fraction of retrieved entities that match ground-truth labels."""
    if not retrieved_ids:
        return 0.0
    expected = {normalize_eval_label(label) for label in relevant_labels}
    relevant_count = 0
    for entity_id in retrieved_ids:
        entity = kg.entities.get(entity_id)
        if entity is not None and entity_label_set(entity) & expected:
            relevant_count += 1
    return relevant_count / len(retrieved_ids)


def exact_context_recall(
    kg: KnowledgeGraph,
    retrieved_ids: list[str],
    relevant_labels: list[str],
) -> float:
    """Return fraction of ground-truth labels covered by retrieved entities."""
    expected = {normalize_eval_label(label) for label in relevant_labels}
    if not expected:
        return 1.0
    found: set[str] = set()
    for entity_id in retrieved_ids:
        entity = kg.entities.get(entity_id)
        if entity is not None:
            found.update(entity_label_set(entity) & expected)
    return len(found) / len(expected)


def content_tokens(text: str) -> set[str]:
    """Tokenize text for lightweight lexical evaluation."""
    normalized = normalize_eval_label(text)
    tokens = set(re.findall(r"[a-zа-я0-9]+\+?", normalized))
    return {token for token in tokens if token not in _STOPWORDS and len(token) > 1}


def save_generation_eval(results: dict[str, Any], out_dir: str | Path) -> None:
    """Save generation summary and per-question metrics."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generation_metrics.json").write_text(
        json.dumps(results["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "generation_metrics_per_question.json").write_text(
        json.dumps(results["per_question"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_generation_summary(results: dict[str, Any]) -> None:
    """Print a compact generation metric table."""
    print("Generation metrics")
    for metric, value in results["summary"].items():
        print(f"{metric:<20} {value:.3f}")


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "как",
    "что",
    "при",
    "для",
    "или",
    "это",
    "по",
    "на",
    "в",
    "и",
}


def main() -> None:
    """CLI entry point for generation evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run generation evaluation.")
    parser.add_argument("--kg", default=str(DEFAULT_KG_PATH))
    parser.add_argument("--gt", default=str(DEFAULT_GT_PATH))
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    results = run_generation_eval(
        kg_path=args.kg,
        gt_path=args.gt,
        top_k=args.top_k,
        n_questions=args.n,
    )
    print_generation_summary(results)
    save_generation_eval(results, args.out_dir)


if __name__ == "__main__":
    main()
