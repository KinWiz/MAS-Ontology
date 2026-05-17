"""End-to-end demo pipeline orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from hirag_ontology.config import load_gemma_settings
from hirag_ontology.llm import GemmaOllamaClient, LLMClient
from hirag_ontology.ontology import load_ontology
from hirag_ontology.pipeline.chunking import load_markdown_chunks
from hirag_ontology.pipeline.deduplication import DeduplicationAgent
from hirag_ontology.pipeline.extractor import ExtractionAgent
from hirag_ontology.pipeline.knowledge_graph import KnowledgeGraph
from hirag_ontology.pipeline.reasoning import ReasoningAgent
from hirag_ontology.pipeline.typing_agent import TypingAgent
from hirag_ontology.pipeline.validator import ValidationAgent
from hirag_ontology.retrieval.retriever import (
    FakeEmbeddingProvider,
    HybridRetriever,
    RetrievalMode,
)

DEMO_QUERY = "How is Ph+ ALL managed differently?"
logger = logging.getLogger(__name__)


def run_demo_pipeline(
    *,
    input_dir: str | Path,
    out_path: str | Path,
    llm: str = "gemma",
    chunk_size: int = 800,
    overlap: int = 100,
) -> dict[str, Any]:
    """Run the MVP demo pipeline."""
    if llm != "gemma":
        msg = "llm must be: gemma"
        raise ValueError(msg)

    input_path = _resolve_input_dir(input_dir)
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_root = output_path.parent / ".cache" / "hirag_demo" / llm

    ontology = load_ontology()
    chunks = load_markdown_chunks(
        input_path,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if not chunks:
        msg = f"No Markdown chunks loaded from {input_path}"
        raise ValueError(msg)

    kg = KnowledgeGraph()
    extraction_client = _build_extraction_client(llm)
    extractor = ExtractionAgent(
        extraction_client,
        ontology=ontology,
        cache_dir=cache_root / "extraction",
    )
    for chunk in chunks:
        extractor.extract_to_graph(kg, chunk)

    raw_entity_count = len(kg.entities)
    raw_relation_count = len(kg.relations)

    typing_stats = TypingAgent(
        _build_typing_client(llm, extraction_client),
        ontology=ontology,
        cache_dir=cache_root / "typing",
    ).type_graph(kg)

    dedup_result = DeduplicationAgent().deduplicate(kg)

    validator = ValidationAgent(ontology)
    validation_before = validator.validate(kg)
    repair_stats = validator.auto_repair(kg, validation_before)
    validation_after_repair = validator.validate(kg)

    reasoning_stats = ReasoningAgent().apply(kg)
    validation_final = validator.validate(kg)

    kg.compute_pagerank()
    kg.save(output_path)

    retrieved = HybridRetriever(
        kg,
        demo_embedding_provider(),
        mode=RetrievalMode.HYBRID_RRF,
    ).retrieve(DEMO_QUERY, top_k=5)

    summary_path = output_path.with_name("run_summary.json")
    summary = {
        "documents_processed": len({chunk.document_id for chunk in chunks}),
        "chunks_processed": len(chunks),
        "entity_count_raw": raw_entity_count,
        "relation_count_raw": raw_relation_count,
        "entity_count_final": len(kg.entities),
        "relation_count_final": len(kg.relations),
        "dedup_merged_count": dedup_result.merged_count,
        "typing": typing_stats,
        "consistency_before": validation_before["consistency_score"],
        "consistency_after_repair": validation_after_repair["consistency_score"],
        "consistency_final": validation_final["consistency_score"],
        "repair": repair_stats,
        "reasoning": reasoning_stats,
        "pagerank_computed": bool(kg.pagerank),
        "graph_path": str(output_path),
        "summary_path": str(summary_path),
        "retrieval_query": DEMO_QUERY,
        "retrieved_entities": [
            {
                "rank": result.rank,
                "entity_id": result.entity_id,
                "label": result.entity.label,
                "entity_type": result.entity.entity_type,
                "score": result.score,
            }
            for result in retrieved
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Demo graph saved to %s", output_path)
    logger.info("Demo run summary saved to %s", summary_path)
    return summary


def _build_extraction_client(llm: str) -> LLMClient:
    settings = load_gemma_settings()
    logger.info(
        "Using local Gemma 4 runtime: model=%s base_url=%s",
        settings.model,
        settings.base_url,
    )
    return GemmaOllamaClient(
        model=settings.model,
        base_url=settings.base_url,
        temperature=settings.temperature,
        max_retries=settings.max_retries,
        min_request_interval_seconds=settings.min_request_interval_seconds,
        request_timeout_seconds=settings.request_timeout_seconds,
    )


def _build_typing_client(llm: str, extraction_client: LLMClient) -> LLMClient:
    del llm
    return extraction_client


def _resolve_input_dir(input_dir: str | Path) -> Path:
    input_path = Path(input_dir)
    if input_path.is_absolute() or input_path.exists():
        return input_path

    project_relative_path = _project_root() / input_path
    if project_relative_path.exists():
        return project_relative_path

    return input_path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def apply_shared_condition_reasoning(kg: KnowledgeGraph) -> dict[str, Any]:
    """Backward-compatible wrapper for the old runner-local reasoning function."""
    return ReasoningAgent().apply(kg)


def demo_embedding_provider() -> FakeEmbeddingProvider:
    """Return deterministic embeddings tuned for the sample demo graph."""
    return FakeEmbeddingProvider(
        {
            "ph+ all": [1.0, 0.0, 0.0],
            "how is ph+ all managed differently?": [1.0, 0.0, 0.0],
            "ph+ acute lymphoblastic leukemia": [1.0, 0.0, 0.0],
            "imatinib": [0.95, 0.05, 0.0],
            "dasatinib": [0.9, 0.1, 0.0],
            "rt-pcr": [0.7, 0.3, 0.0],
            "fish": [0.65, 0.35, 0.0],
            "400 mg daily": [0.3, 0.0, 0.7],
            "nausea": [0.2, 0.8, 0.0],
            "induction therapy": [0.5, 0.5, 0.0],
            "all treatment protocol": [0.55, 0.45, 0.0],
        }
    )
