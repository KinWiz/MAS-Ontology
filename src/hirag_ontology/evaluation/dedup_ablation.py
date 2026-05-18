"""Deduplication hyperparameter ablation over alpha and threshold grids."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from hirag_ontology.evaluation.retrieval_eval import (
    DEFAULT_KG_PATH,
    normalize_eval_label,
)
from hirag_ontology.ontology import load_ontology
from hirag_ontology.pipeline.deduplication import DeduplicationAgent
from hirag_ontology.pipeline.knowledge_graph import Entity, KnowledgeGraph
from hirag_ontology.pipeline.validator import ValidationAgent


@dataclass(frozen=True)
class LabeledPair:
    """One manually labelled duplicate/non-duplicate pair."""

    left: str
    right: str
    is_duplicate: bool


DEFAULT_PAIRS: tuple[LabeledPair, ...] = (
    LabeledPair("ОЛЛ", "Острый лимфобластный лейкоз (ОЛЛ)", True),
    LabeledPair("ОПЛ", "Острый промиелоцитарный лейкоз", True),
    LabeledPair("ТГСК", "алло-ТГСК", True),
    LabeledPair("ПЭТ/КТ", "ПЭТ-КТ", True),
    LabeledPair("FISH", "FISH-исследование", True),
    LabeledPair("ингибиторы тирозинкиназы BCR-ABL", "ТКИ", True),
    LabeledPair("химиотерапия", "ХТ", True),
    LabeledPair("метотрексат", "MTX", True),
    LabeledPair("ОЛЛ", "ОМЛ", False),
    LabeledPair("ОПЛ", "ОЛЛ", False),
    LabeledPair("митотан", "метотрексат", False),
    LabeledPair("цитарабин", "цисплатин", False),
    LabeledPair("лучевая терапия", "химиотерапия", False),
    LabeledPair("ПЭТ/КТ", "FISH", False),
    LabeledPair("венетоклакс", "дексаметазон", False),
    LabeledPair("R-CHOP", "CHOP", False),
)


def evaluate_pairs(
    kg: KnowledgeGraph,
    *,
    alpha: float,
    threshold: float,
    pairs: tuple[LabeledPair, ...] = DEFAULT_PAIRS,
) -> dict[str, Any]:
    """Evaluate whether a deduplication configuration merges labelled pairs."""
    agent = DeduplicationAgent(alpha=alpha, threshold=threshold)
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for pair in pairs:
        left = _find_entity(kg, pair.left) or Entity(label=pair.left)
        right = _find_entity(kg, pair.right) or Entity(label=pair.right)
        predicted_duplicate = agent.hybrid_similarity(left, right) >= threshold
        if pair.is_duplicate and predicted_duplicate:
            true_positive += 1
        elif pair.is_duplicate and not predicted_duplicate:
            false_negative += 1
        elif not pair.is_duplicate and predicted_duplicate:
            false_positive += 1
        else:
            true_negative += 1

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 1.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "n_pairs": len(pairs),
    }


def run_dedup_ablation(
    *,
    kg_path: str | Path = DEFAULT_KG_PATH,
    alphas: list[float] | None = None,
    thresholds: list[float] | None = None,
    pairs: tuple[LabeledPair, ...] = DEFAULT_PAIRS,
    apply_graph_dedup: bool = False,
) -> dict[str, Any]:
    """Run a grid search over deduplication alpha and threshold."""
    started = time.perf_counter()
    kg = KnowledgeGraph.load(kg_path)
    selected_alphas = alphas or [0.4, 0.5, 0.6, 0.7, 0.8]
    selected_thresholds = thresholds or [0.75, 0.8, 0.85, 0.9, 0.95]
    validator = ValidationAgent(load_ontology())
    rows: list[dict[str, Any]] = []

    for alpha, threshold in product(selected_alphas, selected_thresholds):
        row: dict[str, Any] = {
            "alpha": alpha,
            "threshold": threshold,
            **evaluate_pairs(
                kg,
                alpha=alpha,
                threshold=threshold,
                pairs=pairs,
            ),
        }
        if apply_graph_dedup:
            kg_copy = copy.deepcopy(kg)
            before = len(kg_copy.entities)
            dedup_started = time.perf_counter()
            result = DeduplicationAgent(
                alpha=alpha,
                threshold=threshold,
            ).deduplicate(kg_copy)
            validation = validator.validate(kg_copy)
            row.update(
                {
                    "entity_count_after": len(kg_copy.entities),
                    "dedup_rate": (
                        (before - len(kg_copy.entities)) / before
                        if before
                        else 0.0
                    ),
                    "merged_count": result.merged_count,
                    "consistency": validation["consistency_score"],
                    "dedup_elapsed_s": time.perf_counter() - dedup_started,
                }
            )
        rows.append(row)

    rows.sort(key=lambda item: (-float(item["f1"]), -float(item["precision"])))
    return {
        "kg_path": str(kg_path),
        "elapsed_s": time.perf_counter() - started,
        "apply_graph_dedup": apply_graph_dedup,
        "best": rows[0] if rows else None,
        "results": rows,
    }


def save_dedup_ablation(results: dict[str, Any], out_dir: str | Path) -> None:
    """Save dedup ablation results."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dedup_ablation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_dedup_summary(results: dict[str, Any]) -> None:
    """Print the top dedup ablation configurations."""
    print("Deduplication ablation")
    print("alpha threshold precision recall f1")
    for row in results["results"][:10]:
        print(
            f"{row['alpha']:<5.1f} "
            f"{row['threshold']:<9.2f} "
            f"{row['precision']:<9.3f} "
            f"{row['recall']:<6.3f} "
            f"{row['f1']:<6.3f}"
        )


def _find_entity(kg: KnowledgeGraph, label: str) -> Entity | None:
    target = normalize_eval_label(label)
    for entity in kg.entities.values():
        if target in {normalize_eval_label(entity.label), *[
            normalize_eval_label(alias) for alias in entity.aliases
        ]}:
            return entity
    return None


def main() -> None:
    """CLI entry point for dedup ablation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run deduplication ablation.")
    parser.add_argument("--kg", default=str(DEFAULT_KG_PATH))
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--apply-graph-dedup", action="store_true")
    args = parser.parse_args()

    results = run_dedup_ablation(
        kg_path=args.kg,
        apply_graph_dedup=args.apply_graph_dedup,
    )
    print_dedup_summary(results)
    save_dedup_ablation(results, args.out_dir)


if __name__ == "__main__":
    main()
