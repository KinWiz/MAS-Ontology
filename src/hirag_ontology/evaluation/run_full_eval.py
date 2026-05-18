"""Run the full evaluation suite and save consolidated reports."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hirag_ontology.evaluation.baseline_eval import (
    print_baseline_summary,
    run_baseline_eval,
    save_baseline_eval,
)
from hirag_ontology.evaluation.dedup_ablation import (
    print_dedup_summary,
    run_dedup_ablation,
    save_dedup_ablation,
)
from hirag_ontology.evaluation.generation_eval import (
    print_generation_summary,
    run_generation_eval,
    save_generation_eval,
)
from hirag_ontology.evaluation.latency_eval import (
    print_latency_summary,
    run_latency_eval,
    save_latency_eval,
)
from hirag_ontology.evaluation.retrieval_eval import (
    DEFAULT_GT_PATH,
    DEFAULT_KG_PATH,
    print_retrieval_summary,
    run_retrieval_eval,
    save_retrieval_eval,
)
from hirag_ontology.llm import LLMClient
from hirag_ontology.retrieval.retriever import EmbeddingProvider, RetrievalMode

MARKDOWN_REPORT_NAME = "evaluation_report.md"


def run_full_evaluation(
    *,
    kg_path: str | Path = DEFAULT_KG_PATH,
    gt_path: str | Path = DEFAULT_GT_PATH,
    out_dir: str | Path = "results",
    top_k: int = 10,
    n_latency: int = 20,
    n_generation: int | None = None,
    answer_mode: str = "deterministic",
    llm_client: LLMClient | None = None,
    judge_client: LLMClient | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    skip_generation: bool = False,
    skip_dedup: bool = False,
    skip_baselines: bool = False,
    apply_dedup_ablation: bool = False,
) -> dict[str, Any]:
    """Run retrieval, generation, latency, and dedup ablation evaluations."""
    started = time.perf_counter()
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "kg_path": str(kg_path),
        "gt_path": str(gt_path),
        "top_k": top_k,
        "components": {},
    }

    retrieval = run_retrieval_eval(
        kg_path=kg_path,
        gt_path=gt_path,
        top_k=top_k,
        embedding_provider=embedding_provider,
    )
    save_retrieval_eval(retrieval, output_dir)
    report["components"]["retrieval"] = retrieval["per_mode"]

    if not skip_baselines:
        baselines = run_baseline_eval(
            kg_path=kg_path,
            gt_path=gt_path,
            top_k=top_k,
            embedding_provider=embedding_provider,
        )
        save_baseline_eval(baselines, output_dir)
        report["components"]["baselines"] = baselines["per_system"]

    if not skip_generation:
        generation = run_generation_eval(
            kg_path=kg_path,
            gt_path=gt_path,
            top_k=top_k,
            retrieval_mode=RetrievalMode.LEXICAL_STRUCTURAL,
            answer_mode=answer_mode,
            llm_client=llm_client,
            judge_client=judge_client,
            embedding_provider=embedding_provider,
            n_questions=n_generation,
        )
        save_generation_eval(generation, output_dir)
        report["components"]["generation"] = generation["summary"]

    latency = run_latency_eval(
        kg_path=kg_path,
        gt_path=gt_path,
        n_queries=n_latency,
        top_k=top_k,
        answer_mode=answer_mode,
        llm_client=llm_client,
        embedding_provider=embedding_provider,
    )
    save_latency_eval(latency, output_dir)
    report["components"]["latency"] = latency["aggregated"]

    if not skip_dedup:
        dedup = run_dedup_ablation(
            kg_path=kg_path,
            apply_graph_dedup=apply_dedup_ablation,
        )
        save_dedup_ablation(dedup, output_dir)
        report["components"]["dedup_ablation"] = {
            "best": dedup["best"],
            "apply_graph_dedup": apply_dedup_ablation,
        }

    report["total_elapsed_s"] = round(time.perf_counter() - started, 4)
    (output_dir / "full_evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_markdown_report(report, output_dir)
    return report


def save_markdown_report(report: dict[str, Any], out_dir: str | Path) -> Path:
    """Save a compact Markdown report for thesis-style inspection."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MARKDOWN_REPORT_NAME
    lines = [
        "# HiRAG-Ontology Evaluation Report",
        "",
        f"- timestamp: `{report['timestamp']}`",
        f"- graph: `{report['kg_path']}`",
        f"- benchmark: `{report['gt_path']}`",
        f"- top_k: `{report['top_k']}`",
        f"- total_elapsed_s: `{report['total_elapsed_s']}`",
        "",
    ]
    components = report["components"]
    retrieval = components.get("retrieval")
    if isinstance(retrieval, dict):
        lines.extend(
            [
                "## Retrieval",
                "",
                "| mode | Hit@5 | Hit@10 | MRR | MAP@10 | n |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for mode, metrics in retrieval.items():
            if not isinstance(metrics, dict):
                continue
            lines.append(
                "| "
                f"{mode} | "
                f"{_fmt(metrics.get('Hit@5'))} | "
                f"{_fmt(metrics.get('Hit@10'))} | "
                f"{_fmt(metrics.get('MRR'))} | "
                f"{_fmt(metrics.get('MAP@10'))} | "
                f"{metrics.get('n_questions', '')} |"
            )
        lines.append("")

    baselines = components.get("baselines")
    if isinstance(baselines, dict):
        lines.extend(
            [
                "## Baselines",
                "",
                "| system | Hit@5 | Hit@10 | MRR | MAP@10 | n |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for system, metrics in baselines.items():
            if not isinstance(metrics, dict):
                continue
            lines.append(
                "| "
                f"{system} | "
                f"{_fmt(metrics.get('Hit@5'))} | "
                f"{_fmt(metrics.get('Hit@10'))} | "
                f"{_fmt(metrics.get('MRR'))} | "
                f"{_fmt(metrics.get('MAP@10'))} | "
                f"{metrics.get('n_questions', '')} |"
            )
        lines.append("")

    generation = components.get("generation")
    if isinstance(generation, dict):
        lines.extend(
            [
                "## Generation",
                "",
                "| metric | value |",
                "| --- | ---: |",
            ]
        )
        for metric, value in generation.items():
            lines.append(f"| {metric} | {_fmt(value)} |")
        lines.append("")

    latency = components.get("latency")
    if isinstance(latency, dict):
        lines.extend(
            [
                "## Latency",
                "",
                "| stage | mean | std | min | max |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for stage, metrics in latency.items():
            if not isinstance(metrics, dict):
                continue
            lines.append(
                "| "
                f"{stage} | "
                f"{_fmt(metrics.get('mean'))} | "
                f"{_fmt(metrics.get('std'))} | "
                f"{_fmt(metrics.get('min'))} | "
                f"{_fmt(metrics.get('max'))} |"
            )
        lines.append("")

    dedup = components.get("dedup_ablation")
    if isinstance(dedup, dict) and isinstance(dedup.get("best"), dict):
        best = dedup["best"]
        lines.extend(
            [
                "## Deduplication Ablation",
                "",
                "| alpha | threshold | precision | recall | f1 |",
                "| ---: | ---: | ---: | ---: | ---: |",
                "| "
                f"{best.get('alpha', '')} | "
                f"{best.get('threshold', '')} | "
                f"{_fmt(best.get('precision'))} | "
                f"{_fmt(best.get('recall'))} | "
                f"{_fmt(best.get('f1'))} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- Generation metrics are deterministic unless an explicit LLM "
            "client is passed.",
            "- Remote LLM APIs are not called by the default evaluation command.",
            "- Treat these metrics as reproducible MVP metrics, not clinical "
            "validation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def print_full_report(report: dict[str, Any]) -> None:
    """Print the consolidated evaluation report."""
    print("Full evaluation report")
    print(f"kg: {report['kg_path']}")
    print(f"gt: {report['gt_path']}")
    components = report["components"]
    if "retrieval" in components:
        print_retrieval_summary({"per_mode": components["retrieval"]})
    if "baselines" in components:
        print_baseline_summary({"per_system": components["baselines"]})
    if "generation" in components:
        print_generation_summary({"summary": components["generation"]})
    if "latency" in components:
        print_latency_summary({"aggregated": components["latency"]})
    if "dedup_ablation" in components:
        print_dedup_summary({"results": [components["dedup_ablation"]["best"]]})


def _fmt(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    """CLI entry point for the full evaluation suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Run full evaluation suite.")
    parser.add_argument("--kg", default=str(DEFAULT_KG_PATH))
    parser.add_argument("--gt", default=str(DEFAULT_GT_PATH))
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-latency", type=int, default=20)
    parser.add_argument("--n-generation", type=int, default=None)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-dedup", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--apply-dedup-ablation", action="store_true")
    args = parser.parse_args()

    report = run_full_evaluation(
        kg_path=args.kg,
        gt_path=args.gt,
        out_dir=args.out_dir,
        top_k=args.top_k,
        n_latency=args.n_latency,
        n_generation=args.n_generation,
        skip_generation=args.skip_generation,
        skip_dedup=args.skip_dedup,
        skip_baselines=args.skip_baselines,
        apply_dedup_ablation=args.apply_dedup_ablation,
    )
    print_full_report(report)


if __name__ == "__main__":
    main()
