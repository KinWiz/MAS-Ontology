"""Command line entry points for the HiRAG-Ontology prototype."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from hirag_ontology import __version__
from hirag_ontology.config import load_gemma_settings
from hirag_ontology.llm import GemmaOllamaClient
from hirag_ontology.pipeline.knowledge_graph import KnowledgeGraph
from hirag_ontology.pipeline.runner import demo_embedding_provider, run_demo_pipeline
from hirag_ontology.retrieval.answering import (
    answer_from_graph_context,
    build_graph_context,
)
from hirag_ontology.retrieval.retriever import HybridRetriever, RetrievalMode


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
        choices=("gemma",),
        default="gemma",
        help="LLM backend to use. Gemma mode uses local Ollama.",
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
        choices=("gemma",),
        default="gemma",
        help="Answer backend to use. Gemma mode uses local Ollama.",
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
        default=RetrievalMode.HYBRID_RRF.value,
        help="Retrieval mode for selecting graph entities.",
    )
    ask.add_argument(
        "--show-context",
        action="store_true",
        help="Print the graph context used for answering.",
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
            demo_embedding_provider(),
            mode=RetrievalMode(args.retrieval_mode),
        ).retrieve(args.query, top_k=args.top_k)
        graph_context = build_graph_context(kg, retrieved)
        answer = answer_from_graph_context(
            _build_gemma_answer_client(),
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


def _build_gemma_answer_client() -> GemmaOllamaClient:
    settings = load_gemma_settings()
    return GemmaOllamaClient(
        model=settings.model,
        base_url=settings.base_url,
        temperature=settings.temperature,
        max_retries=settings.max_retries,
        min_request_interval_seconds=settings.min_request_interval_seconds,
        request_timeout_seconds=settings.request_timeout_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-demo":
        return run_demo(args)
    if args.command == "ask":
        return run_ask(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
