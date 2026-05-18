"""Intrinsic retrieval evaluation: Hit@K, MRR, and MAP@K."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from hirag_ontology.evaluation.benchmark import load_benchmark
from hirag_ontology.pipeline.knowledge_graph import (
    Entity,
    KnowledgeGraph,
    normalize_label,
)
from hirag_ontology.pipeline.runner import demo_embedding_provider
from hirag_ontology.retrieval.retriever import (
    EmbeddingProvider,
    HybridRetriever,
    RetrievalMode,
)

DEFAULT_KG_PATH = Path("results") / "knowledge_graph_full_gemma.json"
DEFAULT_GT_PATH = Path("evaluation") / "ground_truth.json"


def normalize_eval_label(label: str) -> str:
    """Normalize labels for exact benchmark matching."""
    return normalize_label(label).replace("ё", "е")


def entity_label_set(entity: Entity) -> set[str]:
    """Return normalized label and aliases for one entity."""
    labels = {normalize_eval_label(entity.label)}
    labels.update(normalize_eval_label(alias) for alias in entity.aliases)
    return {label for label in labels if label}


def retrieved_label_set(entity_ids: list[str], kg: KnowledgeGraph) -> set[str]:
    """Return normalized labels and aliases for retrieved entity IDs."""
    labels: set[str] = set()
    for entity_id in entity_ids:
        entity = kg.entities.get(entity_id)
        if entity is not None:
            labels.update(entity_label_set(entity))
    return labels


def hit_at_k(
    retrieved_ids: list[str],
    kg: KnowledgeGraph,
    relevant_labels: list[str],
    *,
    k: int,
) -> float:
    """Return 1 when any relevant entity appears in the top K."""
    if k <= 0:
        return 0.0
    top_labels = retrieved_label_set(retrieved_ids[:k], kg)
    expected = {normalize_eval_label(label) for label in relevant_labels}
    return 1.0 if top_labels & expected else 0.0


def reciprocal_rank(
    retrieved_ids: list[str],
    kg: KnowledgeGraph,
    relevant_labels: list[str],
) -> float:
    """Return reciprocal rank of the first relevant retrieved entity."""
    expected = {normalize_eval_label(label) for label in relevant_labels}
    for rank, entity_id in enumerate(retrieved_ids, start=1):
        entity = kg.entities.get(entity_id)
        if entity is not None and entity_label_set(entity) & expected:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    retrieved_ids: list[str],
    kg: KnowledgeGraph,
    relevant_labels: list[str],
    *,
    k: int,
) -> float:
    """Return average precision at K for entity-label ground truth."""
    expected = {normalize_eval_label(label) for label in relevant_labels}
    if not expected or k <= 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    matched_expected: set[str] = set()
    for rank, entity_id in enumerate(retrieved_ids[:k], start=1):
        entity = kg.entities.get(entity_id)
        if entity is None:
            continue
        matched = entity_label_set(entity) & expected
        new_matches = matched - matched_expected
        if new_matches:
            hits += 1
            matched_expected.update(new_matches)
            precision_sum += hits / rank

    denominator = min(len(expected), k)
    return precision_sum / denominator if denominator else 0.0


def run_retrieval_eval(
    *,
    kg_path: str | Path = DEFAULT_KG_PATH,
    gt_path: str | Path = DEFAULT_GT_PATH,
    top_k: int = 10,
    modes: list[RetrievalMode] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict[str, Any]:
    """Run retrieval metrics for all configured retrieval modes."""
    started = time.perf_counter()
    kg = KnowledgeGraph.load(kg_path)
    questions = load_benchmark(gt_path)
    selected_modes = modes or list(RetrievalMode)
    provider = embedding_provider or demo_embedding_provider()

    per_mode: dict[str, dict[str, float | int]] = {}
    per_question: list[dict[str, Any]] = []
    accumulators: dict[str, dict[str, list[float]]] = {
        mode.value: {"hit5": [], "hit10": [], "rr": [], "ap10": []}
        for mode in selected_modes
    }

    for question in questions:
        question_log: dict[str, Any] = {
            "id": question.id,
            "question": question.question,
            "type": question.question_type,
            "modes": {},
        }
        for mode in selected_modes:
            retriever = HybridRetriever(kg, provider, mode=mode)
            results = retriever.retrieve(question.question, top_k=top_k)
            ids = [result.entity_id for result in results]
            h5 = hit_at_k(ids, kg, question.relevant_entity_labels, k=5)
            h10 = hit_at_k(ids, kg, question.relevant_entity_labels, k=10)
            rr = reciprocal_rank(ids, kg, question.relevant_entity_labels)
            ap10 = average_precision_at_k(
                ids,
                kg,
                question.relevant_entity_labels,
                k=10,
            )
            accumulators[mode.value]["hit5"].append(h5)
            accumulators[mode.value]["hit10"].append(h10)
            accumulators[mode.value]["rr"].append(rr)
            accumulators[mode.value]["ap10"].append(ap10)
            question_log["modes"][mode.value] = {
                "hit@5": h5,
                "hit@10": h10,
                "rr": rr,
                "ap@10": ap10,
                "top_retrieved": [result.entity.label for result in results[:5]],
            }
        per_question.append(question_log)

    for mode in selected_modes:
        values = accumulators[mode.value]
        per_mode[mode.value] = {
            "Hit@5": _mean(values["hit5"]),
            "Hit@10": _mean(values["hit10"]),
            "MRR": _mean(values["rr"]),
            "MAP@10": _mean(values["ap10"]),
            "n_questions": len(questions),
        }

    return {
        "kg_path": str(kg_path),
        "gt_path": str(gt_path),
        "top_k": top_k,
        "elapsed_s": time.perf_counter() - started,
        "per_mode": per_mode,
        "per_question": per_question,
        "question_type_breakdown": _question_type_breakdown(per_question),
    }


def save_retrieval_eval(results: dict[str, Any], out_dir: str | Path) -> None:
    """Save retrieval summary and per-question metrics."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval_metrics.json").write_text(
        json.dumps(results["per_mode"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "retrieval_metrics_per_question.json").write_text(
        json.dumps(results["per_question"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_retrieval_summary(results: dict[str, Any]) -> None:
    """Print a compact retrieval metric table."""
    print("Retrieval metrics")
    print("mode                 Hit@5   Hit@10  MRR    MAP@10")
    for mode, metrics in results["per_mode"].items():
        print(
            f"{mode:<20} "
            f"{metrics['Hit@5']:.3f}  "
            f"{metrics['Hit@10']:.3f}  "
            f"{metrics['MRR']:.3f}  "
            f"{metrics['MAP@10']:.3f}"
        )


def _question_type_breakdown(per_question: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_question:
        grouped.setdefault(str(row["type"]), []).append(row)

    breakdown: dict[str, Any] = {}
    for question_type, rows in grouped.items():
        modes = rows[0]["modes"].keys() if rows else []
        breakdown[question_type] = {}
        for mode in modes:
            breakdown[question_type][mode] = {
                "n": len(rows),
                "MRR": _mean([float(row["modes"][mode]["rr"]) for row in rows]),
                "Hit@10": _mean(
                    [float(row["modes"][mode]["hit@10"]) for row in rows],
                ),
            }
    return breakdown


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def main() -> None:
    """CLI entry point for retrieval evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run retrieval evaluation.")
    parser.add_argument("--kg", default=str(DEFAULT_KG_PATH))
    parser.add_argument("--gt", default=str(DEFAULT_GT_PATH))
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    results = run_retrieval_eval(
        kg_path=args.kg,
        gt_path=args.gt,
        top_k=args.top_k,
    )
    print_retrieval_summary(results)
    save_retrieval_eval(results, args.out_dir)


if __name__ == "__main__":
    main()
