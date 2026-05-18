"""Command line entry points for the HiRAG-Ontology prototype."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from hirag_ontology import __version__
from hirag_ontology.app.web_demo import run_server
from hirag_ontology.config import load_neo4j_settings
from hirag_ontology.evaluation.run_full_eval import run_full_evaluation
from hirag_ontology.llm import SUPPORTED_LLM_BACKENDS, LLMClient, build_llm_client
from hirag_ontology.ontology import load_ontology
from hirag_ontology.pipeline.graph_repair import GraphRepairOptions, repair_graph
from hirag_ontology.pipeline.knowledge_graph import KnowledgeGraph
from hirag_ontology.pipeline.runner import run_demo_pipeline
from hirag_ontology.retrieval.answering import (
    answer_from_graph_context,
    build_graph_context,
)
from hirag_ontology.retrieval.retriever import (
    HybridRetriever,
    RetrievalMode,
    build_embedding_provider,
)
from hirag_ontology.storage import JsonGraphStore, Neo4jGraphStore


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(
        prog="hirag-ontology",
        description="Research prototype CLI for HiRAG-Ontology.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    run_demo = subparsers.add_parser(
        "run-demo",
        help="Build a demo graph with local Gemma through Ollama.",
    )
    run_demo.add_argument(
        "--input",
        default="data/sample_docs",
        help="Directory containing Markdown sample documents.",
    )
    run_demo.add_argument(
        "--out",
        default="results/demo_graph.json",
        help="Path where the demo graph JSON will be written.",
    )
    run_demo.add_argument(
        "--llm",
        choices=SUPPORTED_LLM_BACKENDS,
        default="gemma",
        help="LLM backend to use for extraction and typing.",
    )

    ask = subparsers.add_parser(
        "ask",
        help="Ask a question against a saved knowledge graph.",
    )
    ask.add_argument(
        "--graph",
        default="results/demo_graph.json",
        help="Path to a saved knowledge graph JSON.",
    )
    ask.add_argument(
        "--query",
        required=True,
        help="User question to answer from the graph context.",
    )
    ask.add_argument(
        "--llm",
        choices=SUPPORTED_LLM_BACKENDS,
        default="gemma",
        help="Answer backend to use.",
    )
    ask.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of entities to retrieve.",
    )
    ask.add_argument(
        "--retrieval-mode",
        choices=tuple(mode.value for mode in RetrievalMode),
        default=RetrievalMode.LEXICAL_STRUCTURAL.value,
        help="Retrieval mode for selecting graph entities.",
    )
    ask.add_argument(
        "--embedding-provider",
        choices=("demo", "auto", "openai", "ollama"),
        default="demo",
        help="Embedding provider for semantic and hybrid retrieval.",
    )
    ask.add_argument(
        "--show-context",
        action="store_true",
        help="Print the graph context used for answering.",
    )

    graph_stats = subparsers.add_parser(
        "graph-stats",
        help="Print statistics for a saved JSON knowledge graph.",
    )
    graph_stats.add_argument(
        "--graph",
        default="results/knowledge_graph_repaired.json",
        help="Path to a saved knowledge graph JSON.",
    )

    repair_graph_parser = subparsers.add_parser(
        "repair-graph",
        help="Repair a saved JSON graph against ontology constraints.",
    )
    repair_graph_parser.add_argument(
        "--graph",
        default="results/knowledge_graph_full_gemma.json",
        help="Path to the source graph JSON.",
    )
    repair_graph_parser.add_argument(
        "--out",
        default="results/knowledge_graph_repaired.json",
        help="Path where the repaired graph JSON will be written.",
    )
    repair_graph_parser.add_argument(
        "--report",
        default=None,
        help="Path where the repair report JSON will be written.",
    )
    repair_graph_parser.add_argument(
        "--infer-types",
        action="store_true",
        help="Infer specific types for entities currently typed as Other.",
    )
    repair_graph_parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep exact duplicate relations after repair.",
    )

    export_neo4j = subparsers.add_parser(
        "export-neo4j",
        help="Export a saved JSON knowledge graph into Neo4j.",
    )
    export_neo4j.add_argument(
        "--graph",
        default="results/knowledge_graph_repaired.json",
        help="Path to a saved knowledge graph JSON.",
    )
    export_neo4j.add_argument(
        "--uri",
        default=None,
        help="Neo4j URI. Defaults to NEO4J_URI or bolt://localhost:7687.",
    )
    export_neo4j.add_argument(
        "--user",
        default=None,
        help="Neo4j user. Defaults to NEO4J_USER or neo4j.",
    )
    export_neo4j.add_argument(
        "--password",
        default=None,
        help="Neo4j password. Defaults to NEO4J_PASSWORD.",
    )
    export_neo4j.add_argument(
        "--database",
        default=None,
        help="Neo4j database. Defaults to NEO4J_DATABASE or neo4j.",
    )
    export_neo4j.add_argument(
        "--clear",
        action="store_true",
        help="Delete existing Neo4j graph data before import.",
    )

    web = subparsers.add_parser(
        "web",
        help="Start the local Web UI.",
    )
    web.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the Web UI server.",
    )
    web.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the Web UI server.",
    )
    web.add_argument(
        "--graph",
        default="results/knowledge_graph_repaired.json",
        help="Default graph JSON opened by the Web UI.",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Run retrieval, generation, latency, and dedup evaluation.",
    )
    evaluate.add_argument(
        "--kg",
        default="results/knowledge_graph_repaired.json",
        help="Path to a saved knowledge graph JSON.",
    )
    evaluate.add_argument(
        "--gt",
        default="evaluation/ground_truth.json",
        help="Path to benchmark ground-truth JSON.",
    )
    evaluate.add_argument(
        "--out-dir",
        default="results",
        help="Directory where evaluation artifacts will be written.",
    )
    evaluate.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of entities to retrieve per question.",
    )
    evaluate.add_argument(
        "--n-latency",
        type=int,
        default=20,
        help="Number of questions used for latency evaluation.",
    )
    evaluate.add_argument(
        "--n-generation",
        type=int,
        default=None,
        help="Optional number of questions used for generation evaluation.",
    )
    evaluate.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip answer-generation metrics.",
    )
    evaluate.add_argument(
        "--skip-dedup",
        action="store_true",
        help="Skip deduplication ablation.",
    )
    evaluate.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip Naive RAG vs HiRAG vs HiRAG-Ontology comparison.",
    )
    evaluate.add_argument(
        "--embedding-provider",
        choices=("demo", "auto", "openai", "ollama"),
        default="demo",
        help="Embedding provider for semantic and hybrid retrieval.",
    )
    evaluate.add_argument(
        "--apply-dedup-ablation",
        action="store_true",
        help="Apply each dedup config to a graph copy during ablation.",
    )

    return parser


def run_demo(args: argparse.Namespace) -> int:
    """Run the deterministic demo pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    try:
        summary = run_demo_pipeline(
            input_dir=args.input,
            out_path=args.out,
            llm=args.llm,
        )
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 2

    print(f"Demo graph saved: {summary['graph_path']}")
    print(f"Run summary saved: {summary['summary_path']}")
    print(
        "Graph: "
        f"{summary['entity_count_final']} entities, "
        f"{summary['relation_count_final']} relations"
    )
    print(
        "Top retrieval: "
        + ", ".join(item["label"] for item in summary["retrieved_entities"][:3])
    )
    return 0


def run_ask(args: argparse.Namespace) -> int:
    """Answer a user question against a saved graph."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    try:
        kg = KnowledgeGraph.load(args.graph)
        retrieved = HybridRetriever(
            kg,
            build_embedding_provider(args.embedding_provider),
            mode=RetrievalMode(args.retrieval_mode),
        ).retrieve(args.query, top_k=args.top_k)
        graph_context = build_graph_context(kg, retrieved, query=args.query)
        answer = answer_from_graph_context(
            _build_answer_client(args.llm),
            query=args.query,
            graph_context=graph_context,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 2

    print("Answer:")
    print(answer)
    print("")
    print("Retrieved entities:")
    for result in retrieved:
        print(f"{result.rank}. {result.entity.label} [{result.entity.entity_type}]")
    if args.show_context:
        print("")
        print("Graph context:")
        print(graph_context)
    return 0


def run_graph_stats(args: argparse.Namespace) -> int:
    """Print statistics for a JSON graph."""
    try:
        stats = JsonGraphStore(args.graph).stats()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 2

    print(f"Graph: {stats.entity_count} entities, {stats.relation_count} relations")
    return 0


def run_repair_graph(args: argparse.Namespace) -> int:
    """Repair a JSON graph against ontology constraints."""
    try:
        kg = KnowledgeGraph.load(args.graph)
        report = repair_graph(
            kg,
            ontology=load_ontology(),
            options=GraphRepairOptions(
                infer_other_types=args.infer_types,
                deduplicate_relations=not args.keep_duplicates,
            ),
        )
        output_path = Path(args.out)
        kg.save(output_path)
        report_path = (
            Path(args.report)
            if args.report is not None
            else output_path.with_suffix(".repair_report.json")
        )
        report["input_graph"] = str(args.graph)
        report["output_graph"] = str(output_path)
        report["report_path"] = str(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 2

    before = report["validation_before"]
    after = report["validation_after"]
    print(f"Repaired graph saved: {output_path}")
    print(f"Repair report saved: {report_path}")
    print(
        "Violations: "
        f"{before['violation_count']} -> {after['violation_count']}"
    )
    print(
        "Consistency: "
        f"{before['consistency_score']:.4f} -> {after['consistency_score']:.4f}"
    )
    print(
        "Relations: "
        f"{report['relation_count_before']} -> {report['relation_count_after']}"
    )
    return 0


def run_export_neo4j(args: argparse.Namespace) -> int:
    """Export a JSON graph into an optional Neo4j backend."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    try:
        kg = KnowledgeGraph.load(args.graph)
        settings = load_neo4j_settings()
        uri = args.uri or settings.uri
        user = args.user or settings.user
        password = args.password if args.password is not None else settings.password
        database = args.database if args.database is not None else settings.database
        if not password:
            print("error: Neo4j password is required via --password or NEO4J_PASSWORD")
            return 2

        store = Neo4jGraphStore(
            uri=uri,
            user=user,
            password=password,
            database=database,
        )
        try:
            store.write_graph(kg, clear=args.clear)
            stats = store.stats()
        finally:
            store.close()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 2

    print(
        "Neo4j export complete: "
        f"{stats.entity_count} entities, {stats.relation_count} relations"
    )
    print(f"Neo4j target: {uri} database={database or '<default>'} user={user}")
    return 0


def run_web(args: argparse.Namespace) -> int:
    """Start the local Web UI server."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_server(host=args.host, port=args.port, graph_path=args.graph)
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    """Run the full deterministic evaluation suite."""
    try:
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
            embedding_provider=build_embedding_provider(args.embedding_provider),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 2

    print(f"Evaluation report saved: {args.out_dir}/full_evaluation_report.json")
    print(f"Components: {', '.join(report['components'])}")
    return 0


def _build_answer_client(llm: str) -> LLMClient:
    if llm == "gemma":
        return _build_gemma_answer_client()
    return build_llm_client(llm)


def _build_gemma_answer_client() -> LLMClient:
    return build_llm_client("gemma")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-demo":
        return run_demo(args)
    if args.command == "ask":
        return run_ask(args)
    if args.command == "graph-stats":
        return run_graph_stats(args)
    if args.command == "repair-graph":
        return run_repair_graph(args)
    if args.command == "export-neo4j":
        return run_export_neo4j(args)
    if args.command == "web":
        return run_web(args)
    if args.command == "evaluate":
        return run_evaluate(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
