"""Baseline retrieval comparison for Naive RAG, HiRAG, and HiRAG-Ontology."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hirag_ontology.evaluation.retrieval_eval import (
    DEFAULT_GT_PATH,
    DEFAULT_KG_PATH,
    run_retrieval_eval,
)
from hirag_ontology.pipeline.runner import demo_embedding_provider
from hirag_ontology.retrieval.retriever import (
    EmbeddingProvider,
    RetrievalMode,
)

BASELINE_MODES: dict[str, RetrievalMode] = {
    "naive_rag": RetrievalMode.LEXICAL_ONLY,
    "hirag": RetrievalMode.HYBRID_RRF,
    "hirag_ontology": RetrievalMode.LEXICAL_STRUCTURAL,
}


def run_baseline_eval(
    *,
    kg_path: str | Path = DEFAULT_KG_PATH,
    gt_path: str | Path = DEFAULT_GT_PATH,
    top_k: int = 10,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict[str, Any]:
    """Compare simple retrieval baselines on the same benchmark."""
    retrieval = run_retrieval_eval(
        kg_path=kg_path,
        gt_path=gt_path,
        top_k=top_k,
        modes=list(dict.fromkeys(BASELINE_MODES.values())),
        embedding_provider=embedding_provider or demo_embedding_provider(),
    )
    mode_to_baseline = {mode.value: name for name, mode in BASELINE_MODES.items()}
    per_system = {
        baseline: retrieval["per_mode"][mode.value]
        for baseline, mode in BASELINE_MODES.items()
    }
    per_question: list[dict[str, Any]] = []
    for row in retrieval["per_question"]:
        per_question.append(
            {
                "id": row["id"],
                "question": row["question"],
                "type": row["type"],
                "systems": {
                    mode_to_baseline[mode]: metrics
                    for mode, metrics in row["modes"].items()
                    if mode in mode_to_baseline
                },
            }
        )
    return {
        "kg_path": str(kg_path),
        "gt_path": str(gt_path),
        "top_k": top_k,
        "systems": {
            "naive_rag": {
                "label": "Naive RAG",
                "retrieval_mode": RetrievalMode.LEXICAL_ONLY.value,
                "description": "BM25-style lexical entity retrieval.",
            },
            "hirag": {
                "label": "HiRAG",
                "retrieval_mode": RetrievalMode.HYBRID_RRF.value,
                "description": (
                    "RRF fusion over semantic, lexical, and structural ranks."
                ),
            },
            "hirag_ontology": {
                "label": "HiRAG-Ontology",
                "retrieval_mode": RetrievalMode.LEXICAL_STRUCTURAL.value,
                "description": (
                    "Ontology graph-aware lexical retrieval with PageRank signal."
                ),
            },
        },
        "per_system": per_system,
        "per_question": per_question,
    }


def save_baseline_eval(results: dict[str, Any], out_dir: str | Path) -> None:
    """Save baseline summary and per-question comparison artifacts."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(results["per_system"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "baseline_metrics_per_question.json").write_text(
        json.dumps(results["per_question"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_baseline_summary(results: dict[str, Any]) -> None:
    """Print a compact baseline comparison table."""
    print("Baseline comparison")
    print("system               Hit@5   Hit@10  MRR    MAP@10")
    for system, metrics in results["per_system"].items():
        print(
            f"{system:<20} "
            f"{metrics['Hit@5']:.3f}  "
            f"{metrics['Hit@10']:.3f}  "
            f"{metrics['MRR']:.3f}  "
            f"{metrics['MAP@10']:.3f}"
        )
