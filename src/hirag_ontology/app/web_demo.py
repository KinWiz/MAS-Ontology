"""Local Web UI and API for exploring HiRAG-Ontology graphs."""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import networkx as nx

from hirag_ontology.config import load_neo4j_settings
from hirag_ontology.llm import (
    SUPPORTED_ANSWER_BACKENDS,
    SUPPORTED_LLM_BACKENDS,
    LLMClient,
    build_llm_client,
)
from hirag_ontology.ontology import load_ontology
from hirag_ontology.pipeline.chunking import load_markdown_chunks
from hirag_ontology.pipeline.deduplication import DeduplicationAgent
from hirag_ontology.pipeline.extractor import ExtractionAgent
from hirag_ontology.pipeline.knowledge_graph import KnowledgeGraph, Relation
from hirag_ontology.pipeline.reasoning import ReasoningAgent
from hirag_ontology.pipeline.typing_agent import TypingAgent
from hirag_ontology.pipeline.validator import ValidationAgent
from hirag_ontology.retrieval.answering import (
    answer_from_graph_context,
    build_graph_context,
    deterministic_answer_from_graph_context,
)
from hirag_ontology.retrieval.retriever import HybridRetriever, RetrievalMode
from hirag_ontology.storage import Neo4jGraphStore

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).with_name("static")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


@dataclass
class WebAppState:
    """Runtime state shared by Web UI request handlers."""

    default_graph_path: Path


@dataclass
class PipelineStage:
    """One visible Web UI pipeline stage."""

    id: str
    label: str
    status: str = "pending"
    detail: str = ""
    started_at: float | None = None
    completed_at: float | None = None
    duration_s: float | None = None


@dataclass
class PipelineJob:
    """In-memory background pipeline job state."""

    id: str
    status: str
    stages: list[PipelineStage]
    graph_path: str
    llm: str = "gemma"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    summary: dict[str, Any] | None = None
    error: str | None = None


_jobs: dict[str, PipelineJob] = {}
_jobs_lock = threading.Lock()


def run_server(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    graph_path: str | Path | None = None,
) -> None:
    """Start the local Web UI server."""
    server = create_server(host=host, port=port, graph_path=graph_path)
    url = f"http://{host}:{port}"
    print(f"HiRAG-Ontology Web UI: {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("")
    finally:
        server.server_close()


def create_server(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    graph_path: str | Path | None = None,
) -> ThreadingHTTPServer:
    """Create a ThreadingHTTPServer for tests or CLI startup."""
    state = WebAppState(default_graph_path=resolve_project_path(graph_path))

    class WebDemoHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            self._send_empty(HTTPStatus.NO_CONTENT)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_file(STATIC_DIR / "index.html", "text/html")
                    return
                if parsed.path.startswith("/static/"):
                    self._send_static(parsed.path.removeprefix("/static/"))
                    return
                if parsed.path == "/api/dashboard":
                    params = parse_qs(parsed.query)
                    self._send_json(
                        dashboard_payload(
                            _graph_path_from_params(params, state),
                        )
                    )
                    return
                if parsed.path == "/api/entities":
                    params = parse_qs(parsed.query)
                    self._send_json(
                        search_entities_payload(
                            graph_path=_graph_path_from_params(params, state),
                            query=_query_param(params, "query"),
                            entity_type=_query_param(params, "entity_type"),
                            limit=_int_query_param(params, "limit", 30),
                            offset=_int_query_param(params, "offset", 0),
                        )
                    )
                    return
                if parsed.path == "/api/subgraph":
                    params = parse_qs(parsed.query)
                    self._send_json(
                        subgraph_payload(
                            graph_path=_graph_path_from_params(params, state),
                            entity_id=_query_param(params, "entity_id"),
                            depth=_int_query_param(params, "depth", 1),
                            limit_nodes=_int_query_param(params, "limit_nodes", 60),
                            entity_type=_query_param(params, "entity_type"),
                            predicate=_query_param(params, "predicate"),
                        )
                    )
                    return
                if parsed.path.startswith("/api/pipeline/jobs/"):
                    job_id = parsed.path.rsplit("/", 1)[-1]
                    self._send_json(job_payload(job_id))
                    return
                if parsed.path == "/api/neo4j/status":
                    self._send_json(neo4j_status_payload())
                    return
                if parsed.path == "/api/evaluation/summary":
                    self._send_json(evaluation_summary_payload())
                    return
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            except Exception as error:  # pragma: no cover - server safety boundary
                logger.exception("Web UI GET failed")
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                body = self._read_json_body()
                if parsed.path == "/api/ask":
                    self._send_json(
                        ask_payload(
                            graph_path=resolve_project_path(
                                body.get("graph_path"),
                                default=state.default_graph_path,
                            ),
                            query=str(body.get("query", "")),
                            retrieval_mode=str(
                                body.get(
                                    "retrieval_mode",
                                    RetrievalMode.LEXICAL_STRUCTURAL,
                                )
                            ),
                            top_k=_safe_int(body.get("top_k"), default=5),
                            llm=str(body.get("llm", "gemma")),
                        )
                    )
                    return
                if parsed.path == "/api/retrieved-subgraph":
                    self._send_json(
                        retrieved_subgraph_payload(
                            graph_path=resolve_project_path(
                                body.get("graph_path"),
                                default=state.default_graph_path,
                            ),
                            entity_ids=[
                                str(item)
                                for item in body.get("entity_ids", [])
                                if str(item)
                            ],
                            depth=_safe_int(body.get("depth"), default=1),
                            limit_nodes=_safe_int(
                                body.get("limit_nodes"),
                                default=80,
                            ),
                        )
                    )
                    return
                if parsed.path == "/api/retrieval/compare":
                    self._send_json(
                        retrieval_compare_payload(
                            graph_path=resolve_project_path(
                                body.get("graph_path"),
                                default=state.default_graph_path,
                            ),
                            query=str(body.get("query", "")),
                            top_k=_safe_int(body.get("top_k"), default=5),
                        )
                    )
                    return
                if parsed.path == "/api/pipeline/jobs":
                    self._send_json(
                        create_pipeline_job(
                            documents=_documents_from_payload(body),
                            out_path=resolve_project_path(
                                body.get("out_path"),
                                default=_project_root()
                                / "results"
                                / f"web_pipeline_{uuid4().hex[:8]}.json",
                            ),
                            llm=str(body.get("llm", "gemma")),
                        ),
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
                if parsed.path == "/api/neo4j/export":
                    self._send_json(
                        neo4j_export_payload(
                            graph_path=resolve_project_path(
                                body.get("graph_path"),
                                default=state.default_graph_path,
                            ),
                            clear=bool(body.get("clear", False)),
                        )
                    )
                    return
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            except ValueError as error:
                self._send_error(HTTPStatus.BAD_REQUEST, str(error))
            except Exception as error:  # pragma: no cover - server safety boundary
                logger.exception("Web UI POST failed")
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

        def log_message(self, format: str, *args: object) -> None:
            logger.info("%s - %s", self.address_string(), format % args)

        def _read_json_body(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length", "0")
            length = _safe_int(raw_length, default=0)
            if length <= 0:
                return {}
            raw_body = self.rfile.read(length)
            decoded = json.loads(raw_body.decode("utf-8"))
            if not isinstance(decoded, dict):
                msg = "JSON request body must be an object."
                raise ValueError(msg)
            return decoded

        def _send_static(self, relative_path: str) -> None:
            static_root = STATIC_DIR.resolve()
            file_path = (STATIC_DIR / relative_path).resolve()
            if static_root not in file_path.parents:
                self._send_error(HTTPStatus.BAD_REQUEST, "Invalid static path")
                return
            content_type = mimetypes.guess_type(file_path.name)[0] or "text/plain"
            self._send_file(file_path, content_type)

        def _send_file(self, file_path: Path, content_type: str) -> None:
            if not file_path.exists() or not file_path.is_file():
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            payload = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self._send_common_headers(content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(
            self,
            payload: dict[str, Any],
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self._send_common_headers("application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            self._send_json({"error": message}, status=status)

        def _send_empty(self, status: HTTPStatus) -> None:
            self.send_response(status)
            self._send_common_headers("text/plain")
            self.end_headers()

        def _send_common_headers(self, content_type: str) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    return ThreadingHTTPServer((host, port), WebDemoHandler)


def dashboard_payload(graph_path: str | Path) -> dict[str, Any]:
    """Build dashboard statistics for a graph."""
    kg = KnowledgeGraph.load(graph_path)
    if set(kg.pagerank) != set(kg.entities):
        kg.compute_pagerank()
    validation = ValidationAgent(load_ontology()).validate(kg)
    violation_types = Counter(
        str(violation["type"]) for violation in validation["violations"]
    )
    type_distribution = Counter(
        entity.entity_type for entity in kg.entities.values()
    )
    predicate_distribution = Counter(
        relation.predicate for relation in kg.relations
    )

    return {
        "graph_path": str(graph_path),
        "entity_count": len(kg.entities),
        "relation_count": len(kg.relations),
        "type_distribution": dict(sorted(type_distribution.items())),
        "predicate_distribution": dict(sorted(predicate_distribution.items())),
        "graph_metrics": _graph_metrics(kg),
        "top_predicates": [
            {"predicate": predicate, "count": count}
            for predicate, count in predicate_distribution.most_common(10)
        ],
        "top_by_degree": _top_entities_by_degree(kg),
        "top_by_pagerank": _top_entities_by_pagerank(kg),
        "validation": {
            "consistency_score": validation["consistency_score"],
            "violation_count": len(validation["violations"]),
            "violation_types": dict(sorted(violation_types.items())),
            "status": "valid" if not validation["violations"] else "invalid",
        },
        "entity_types": sorted(type_distribution),
        "predicates": sorted({relation.predicate for relation in kg.relations}),
        "retrieval_modes": [mode.value for mode in RetrievalMode],
        "answer_llms": list(SUPPORTED_ANSWER_BACKENDS),
        "pipeline_llms": list(SUPPORTED_LLM_BACKENDS),
    }


def search_entities_payload(
    *,
    graph_path: str | Path,
    query: str = "",
    entity_type: str = "",
    limit: int = 30,
    offset: int = 0,
) -> dict[str, Any]:
    """Search graph entities for the explorer."""
    kg = KnowledgeGraph.load(graph_path)
    normalized_query = query.casefold().strip()
    safe_limit = _clamp(limit, minimum=1, maximum=100)
    safe_offset = max(offset, 0)
    nodes = [
        _entity_payload(kg, entity_id)
        for entity_id, entity in kg.entities.items()
        if _matches_entity(entity, normalized_query)
        and (not entity_type or entity.entity_type == entity_type)
    ]
    nodes.sort(key=lambda node: (str(node["label"]).casefold(), str(node["id"])))
    page = nodes[safe_offset : safe_offset + safe_limit]
    return {
        "items": page,
        "total": len(nodes),
        "limit": safe_limit,
        "offset": safe_offset,
        "has_more": safe_offset + safe_limit < len(nodes),
    }


def subgraph_payload(
    *,
    graph_path: str | Path,
    entity_id: str,
    depth: int = 1,
    limit_nodes: int = 60,
    entity_type: str = "",
    predicate: str = "",
) -> dict[str, Any]:
    """Return a limited graph slice around one entity."""
    kg = KnowledgeGraph.load(graph_path)
    if entity_id not in kg.entities:
        return _empty_subgraph_payload()
    return _build_subgraph_payload(
        kg=kg,
        center_ids=[entity_id],
        depth=depth,
        limit_nodes=limit_nodes,
        entity_type=entity_type,
        predicate=predicate,
    )


def retrieved_subgraph_payload(
    *,
    graph_path: str | Path,
    entity_ids: list[str],
    depth: int = 1,
    limit_nodes: int = 80,
) -> dict[str, Any]:
    """Return a limited graph slice for retrieved answer entities."""
    kg = KnowledgeGraph.load(graph_path)
    center_ids = [entity_id for entity_id in entity_ids if entity_id in kg.entities]
    if not center_ids:
        return _empty_subgraph_payload()
    return _build_subgraph_payload(
        kg=kg,
        center_ids=center_ids,
        depth=depth,
        limit_nodes=limit_nodes,
    )


def ask_payload(
    *,
    graph_path: str | Path,
    query: str,
    retrieval_mode: str,
    top_k: int,
    llm: str = "gemma",
) -> dict[str, Any]:
    """Answer a question using graph retrieval and optional local Gemma."""
    if not query.strip():
        msg = "Question is required."
        raise ValueError(msg)
    kg = KnowledgeGraph.load(graph_path)
    mode = RetrievalMode(retrieval_mode)
    safe_top_k = _clamp(top_k, minimum=1, maximum=20)
    retrieval_started = time.perf_counter()
    retrieved = HybridRetriever(
        kg,
        _web_embedding_provider(),
        mode=mode,
    ).retrieve(query, top_k=safe_top_k)
    retrieval_elapsed = time.perf_counter() - retrieval_started
    graph_context = build_graph_context(kg, retrieved, query=query)

    answer_started = time.perf_counter()
    if llm in SUPPORTED_LLM_BACKENDS:
        answer = answer_from_graph_context(
            _build_chat_client(llm),
            query=query,
            graph_context=graph_context,
        )
    elif llm == "deterministic":
        answer = deterministic_answer_from_graph_context(
            query=query,
            graph_context=graph_context,
            retrieved=retrieved,
        )
    else:
        msg = f"llm must be one of: {', '.join(SUPPORTED_ANSWER_BACKENDS)}."
        raise ValueError(msg)
    answer_elapsed = time.perf_counter() - answer_started
    context_relations = _context_relation_payloads(kg, retrieved, limit=15)
    source_chunks = _source_chunks_from_context(retrieved, context_relations)

    return {
        "answer": answer,
        "graph_context": graph_context,
        "retrieved_entities": [
            _retrieved_entity_payload(result, kg)
            for result in retrieved
        ],
        "context_relations": context_relations,
        "source_chunks": source_chunks,
        "diagnostics": {
            "graph_path": str(graph_path),
            "query": query,
            "retrieval_mode": mode.value,
            "top_k": safe_top_k,
            "llm": llm,
            "retrieved_count": len(retrieved),
            "graph_context_chars": len(graph_context),
            "retrieval_s": retrieval_elapsed,
            "answer_s": answer_elapsed,
            "total_s": retrieval_elapsed + answer_elapsed,
        },
    }


def retrieval_compare_payload(
    *,
    graph_path: str | Path,
    query: str,
    top_k: int,
) -> dict[str, Any]:
    """Compare retrieval results across all available modes."""
    if not query.strip():
        msg = "Question is required."
        raise ValueError(msg)
    kg = KnowledgeGraph.load(graph_path)
    safe_top_k = _clamp(top_k, minimum=1, maximum=20)
    modes: dict[str, Any] = {}
    for mode in RetrievalMode:
        started = time.perf_counter()
        retrieved = HybridRetriever(
            kg,
            _web_embedding_provider(),
            mode=mode,
        ).retrieve(query, top_k=safe_top_k)
        modes[mode.value] = {
            "duration_s": time.perf_counter() - started,
            "items": [
                _retrieved_entity_payload(result, kg)
                for result in retrieved
            ],
        }
    return {
        "query": query,
        "top_k": safe_top_k,
        "modes": modes,
    }


def neo4j_status_payload() -> dict[str, Any]:
    """Return optional Neo4j connection status without exposing secrets."""
    settings = load_neo4j_settings()
    target = {
        "uri": settings.uri,
        "user": settings.user,
        "database": settings.database,
        "password_set": bool(settings.password),
    }
    if not settings.password:
        return {
            "configured": False,
            "connected": False,
            "target": target,
            "message": "NEO4J_PASSWORD is not configured.",
        }

    store = Neo4jGraphStore(
        uri=settings.uri,
        user=settings.user,
        password=settings.password,
        database=settings.database,
    )
    try:
        stats = store.stats()
    except Exception as error:  # pragma: no cover - depends on live Neo4j
        return {
            "configured": True,
            "connected": False,
            "target": target,
            "message": str(error),
        }
    finally:
        store.close()

    return {
        "configured": True,
        "connected": True,
        "target": target,
        "message": "Connected.",
        "entity_count": stats.entity_count,
        "relation_count": stats.relation_count,
    }


def evaluation_summary_payload() -> dict[str, Any]:
    """Load saved evaluation artifacts for the Web UI quality panel."""
    results_dir = _project_root() / "results"
    retrieval_metrics = _read_json_artifact(results_dir / "retrieval_metrics.json")
    generation_metrics = _read_json_artifact(results_dir / "generation_metrics.json")
    latency_metrics = _read_json_artifact(results_dir / "latency_results.json")
    full_report = _read_json_artifact(results_dir / "full_evaluation_report.json")
    return {
        "results_dir": str(results_dir),
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "latency_metrics": latency_metrics,
        "full_report": full_report,
        "has_any_metrics": any(
            item is not None
            for item in [
                retrieval_metrics,
                generation_metrics,
                latency_metrics,
                full_report,
            ]
        ),
    }


def create_pipeline_job(
    *,
    documents: list[dict[str, str]],
    out_path: str | Path,
    llm: str = "gemma",
) -> dict[str, Any]:
    """Create and start a background pipeline job."""
    if not documents:
        msg = "At least one Markdown document is required."
        raise ValueError(msg)
    if llm not in SUPPORTED_LLM_BACKENDS:
        msg = f"llm must be one of: {', '.join(SUPPORTED_LLM_BACKENDS)}."
        raise ValueError(msg)

    job_id = uuid4().hex
    job = PipelineJob(
        id=job_id,
        status="queued",
        stages=_pipeline_stages(),
        graph_path=str(out_path),
        llm=llm,
    )
    with _jobs_lock:
        _jobs[job_id] = job

    thread = threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, documents, Path(out_path), llm),
        daemon=True,
    )
    thread.start()
    return job_payload(job_id)


def job_payload(job_id: str) -> dict[str, Any]:
    """Return a JSON-safe pipeline job payload."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            msg = f"Unknown pipeline job: {job_id}"
            raise ValueError(msg)
        return {
            "id": job.id,
            "status": job.status,
            "graph_path": job.graph_path,
            "llm": job.llm,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "stages": [asdict(stage) for stage in job.stages],
            "summary": job.summary,
            "error": job.error,
        }


def neo4j_export_payload(
    *,
    graph_path: str | Path,
    clear: bool = False,
) -> dict[str, Any]:
    """Export a graph to optional Neo4j storage."""
    kg = KnowledgeGraph.load(graph_path)
    settings = load_neo4j_settings()
    if not settings.password:
        msg = "Neo4j password is required in NEO4J_PASSWORD."
        raise ValueError(msg)

    store = Neo4jGraphStore(
        uri=settings.uri,
        user=settings.user,
        password=settings.password,
        database=settings.database,
    )
    try:
        store.write_graph(kg, clear=clear)
        stats = store.stats()
    finally:
        store.close()

    return {
        "entity_count": stats.entity_count,
        "relation_count": stats.relation_count,
        "target": {
            "uri": settings.uri,
            "user": settings.user,
            "database": settings.database,
        },
    }


def resolve_project_path(
    value: object | None,
    *,
    default: str | Path | None = None,
) -> Path:
    """Resolve user-provided paths against the project root."""
    if value is None or str(value).strip() == "":
        if default is None:
            return _project_root() / "results" / "knowledge_graph_full_gemma.json"
        raw_path = Path(default)
    else:
        raw_path = Path(str(value))

    if raw_path.is_absolute() or raw_path.exists():
        return raw_path
    return _project_root() / raw_path


def _run_pipeline_job(
    job_id: str,
    documents: list[dict[str, str]],
    out_path: Path,
    llm: str,
) -> None:
    _set_job_status(job_id, "running")
    try:
        input_dir = _project_root() / "results" / "web_uploads" / job_id
        input_dir.mkdir(parents=True, exist_ok=True)
        for index, document in enumerate(documents, start=1):
            filename = _safe_markdown_filename(document.get("filename", ""), index)
            (input_dir / filename).write_text(document["content"], encoding="utf-8")

        summary = _run_pipeline_with_progress(
            job_id=job_id,
            input_dir=input_dir,
            out_path=out_path,
            llm=llm,
        )
        _set_job_summary(job_id, summary)
        _set_job_status(job_id, "completed")
    except Exception as error:  # pragma: no cover - background safety boundary
        logger.exception("Pipeline job failed")
        _set_job_error(job_id, f"{error}\n{traceback.format_exc()}")


def _run_pipeline_with_progress(
    *,
    job_id: str,
    input_dir: Path,
    out_path: Path,
    llm: str,
) -> dict[str, Any]:
    ontology = load_ontology()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_root = output_path.parent / ".cache" / "hirag_web" / llm / job_id

    _set_stage(job_id, "A1", "running", "Loading Markdown documents")
    chunks = load_markdown_chunks(input_dir)
    if not chunks:
        msg = f"No Markdown chunks loaded from {input_dir}"
        raise ValueError(msg)
    _set_stage(
        job_id,
        "A1",
        "completed",
        f"{len({chunk.document_id for chunk in chunks})} docs, {len(chunks)} chunks",
    )

    kg = KnowledgeGraph()
    _set_stage(job_id, "A2", "running", "Extracting entities and relations")
    llm_client = _build_chat_client(llm)
    extractor = ExtractionAgent(
        llm_client,
        ontology=ontology,
        cache_dir=cache_root / "extraction",
    )
    for index, chunk in enumerate(chunks, start=1):
        extractor.extract_to_graph(kg, chunk)
        _set_stage(job_id, "A2", "running", f"Chunk {index}/{len(chunks)}")
    raw_entity_count = len(kg.entities)
    raw_relation_count = len(kg.relations)
    _set_stage(
        job_id,
        "A2",
        "completed",
        f"{raw_entity_count} entities, {raw_relation_count} relations",
    )

    _set_stage(job_id, "A3", "running", "Typing entities")
    typing_stats = TypingAgent(
        llm_client,
        ontology=ontology,
        cache_dir=cache_root / "typing",
    ).type_graph(kg)
    _set_stage(job_id, "A3", "completed", f"{typing_stats['typed_count']} typed")

    _set_stage(job_id, "A4", "running", "Deduplicating entities")
    dedup_result = DeduplicationAgent().deduplicate(kg)
    _set_stage(job_id, "A4", "completed", f"{dedup_result.merged_count} merged")

    _set_stage(job_id, "A5", "running", "Validating and reasoning")
    validator = ValidationAgent(ontology)
    validation_before = validator.validate(kg)
    repair_stats = validator.auto_repair(kg, validation_before)
    validation_after_repair = validator.validate(kg)
    reasoning_stats = ReasoningAgent().apply(kg)
    validation_final = validator.validate(kg)
    _set_stage(
        job_id,
        "A5",
        "completed",
        f"{len(validation_final['violations'])} violations",
    )

    _set_stage(job_id, "A6", "running", "Computing PageRank and saving graph")
    kg.compute_pagerank()
    kg.save(output_path)
    retrieved = HybridRetriever(
        kg,
        _web_embedding_provider(),
        mode=RetrievalMode.HYBRID_RRF,
    ).retrieve("How is Ph+ ALL managed differently?", top_k=5)
    summary_path = output_path.with_name("run_summary.json")
    summary = {
        "documents_processed": len({chunk.document_id for chunk in chunks}),
        "llm": llm,
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
        "retrieved_entities": [
            _retrieved_entity_payload(result, kg)
            for result in retrieved
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _set_stage(job_id, "A6", "completed", str(output_path))
    return summary


def _pipeline_stages() -> list[PipelineStage]:
    return [
        PipelineStage("A1", "A1 Ingestion & Chunking"),
        PipelineStage("A2", "A2 Extraction"),
        PipelineStage("A3", "A3 Typing"),
        PipelineStage("A4", "A4 Deduplication"),
        PipelineStage("A5", "A5 Validation & Reasoning"),
        PipelineStage("A6", "A6 PageRank & Save"),
    ]


def _set_job_status(job_id: str, status: str) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job.status = status
        job.updated_at = time.time()


def _set_job_summary(job_id: str, summary: dict[str, Any]) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job.summary = summary
        job.graph_path = str(summary["graph_path"])
        job.updated_at = time.time()


def _set_job_error(job_id: str, error: str) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job.status = "failed"
        job.error = error
        job.updated_at = time.time()
        for stage in job.stages:
            if stage.status == "running":
                stage.status = "failed"


def _set_stage(job_id: str, stage_id: str, status: str, detail: str) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        for stage in job.stages:
            if stage.id == stage_id:
                now = time.time()
                if status == "running" and stage.started_at is None:
                    stage.started_at = now
                if status in {"completed", "failed"}:
                    if stage.started_at is None:
                        stage.started_at = now
                    stage.completed_at = now
                    stage.duration_s = now - stage.started_at
                stage.status = status
                stage.detail = detail
                break
        job.updated_at = time.time()


def _build_chat_client(llm: str) -> LLMClient:
    return build_llm_client(llm)


def _web_embedding_provider() -> Any:
    from hirag_ontology.pipeline.runner import demo_embedding_provider

    return demo_embedding_provider()


def _build_subgraph_payload(
    *,
    kg: KnowledgeGraph,
    center_ids: list[str],
    depth: int,
    limit_nodes: int,
    entity_type: str = "",
    predicate: str = "",
) -> dict[str, Any]:
    safe_depth = _clamp(depth, minimum=1, maximum=3)
    safe_limit = _clamp(limit_nodes, minimum=1, maximum=200)
    center_set = set(center_ids)
    context_ids = set(center_ids)
    for center_id in center_ids:
        context_ids.update(kg.neighbors(center_id, depth=safe_depth))

    allowed_ids = {
        entity_id
        for entity_id in context_ids
        if entity_id in center_set
        or not entity_type
        or kg.entities[entity_id].entity_type == entity_type
    }
    selected_ids = set(center_ids[:safe_limit])
    selected_relations: list[Relation] = []
    for relation in sorted(
        kg.relations,
        key=lambda item: _relation_sort_key(kg, item),
    ):
        if (
            relation.subject_id not in allowed_ids
            or relation.object_id not in allowed_ids
        ):
            continue
        if predicate and relation.predicate != predicate:
            continue
        next_ids = selected_ids | {relation.subject_id, relation.object_id}
        if len(next_ids) > safe_limit:
            continue
        selected_ids = next_ids
        selected_relations.append(relation)

    if not selected_relations and len(selected_ids) < safe_limit:
        neighbors = sorted(
            allowed_ids - selected_ids,
            key=lambda entity_id: kg.entities[entity_id].label.casefold(),
        )
        selected_ids.update(neighbors[: safe_limit - len(selected_ids)])

    return {
        "nodes": [
            _entity_payload(kg, entity_id, selected=entity_id in center_set)
            for entity_id in sorted(
                selected_ids,
                key=lambda item: (
                    item not in center_set,
                    kg.entities[item].label.casefold(),
                    item,
                ),
            )
        ],
        "relations": [
            _relation_payload(kg, relation, index)
            for index, relation in enumerate(selected_relations)
        ],
        "depth": safe_depth,
        "limit_nodes": safe_limit,
    }


def _empty_subgraph_payload() -> dict[str, Any]:
    return {"nodes": [], "relations": [], "depth": 1, "limit_nodes": 0}


def _graph_metrics(kg: KnowledgeGraph) -> dict[str, Any]:
    node_count = len(kg.entities)
    relation_count = len(kg.relations)
    possible_relations = node_count * (node_count - 1)
    weak_components = (
        list(nx.weakly_connected_components(kg.graph))
        if node_count
        else []
    )
    largest_component = max(
        (len(component) for component in weak_components),
        default=0,
    )
    source_chunks = {
        chunk
        for entity in kg.entities.values()
        for chunk in entity.source_chunks
    }
    source_chunks.update(
        relation.source_chunk
        for relation in kg.relations
        if relation.source_chunk
    )
    alias_count = sum(len(entity.aliases) for entity in kg.entities.values())
    return {
        "density": relation_count / possible_relations if possible_relations else 0.0,
        "connected_components": len(weak_components),
        "largest_component_size": largest_component,
        "isolated_entities": len(list(nx.isolates(kg.graph))),
        "source_chunk_count": len(source_chunks),
        "alias_count": alias_count,
        "pagerank_available": bool(kg.pagerank),
    }


def _top_entities_by_degree(
    kg: KnowledgeGraph,
    limit: int = 10,
) -> list[dict[str, Any]]:
    entity_ids = sorted(
        kg.entities,
        key=lambda entity_id: (
            -int(kg.graph.degree[entity_id]),
            kg.entities[entity_id].label.casefold(),
            entity_id,
        ),
    )
    return [_entity_payload(kg, entity_id) for entity_id in entity_ids[:limit]]


def _top_entities_by_pagerank(
    kg: KnowledgeGraph,
    limit: int = 10,
) -> list[dict[str, Any]]:
    entity_ids = sorted(
        kg.entities,
        key=lambda entity_id: (
            -kg.pagerank.get(entity_id, 0.0),
            kg.entities[entity_id].label.casefold(),
            entity_id,
        ),
    )
    return [_entity_payload(kg, entity_id) for entity_id in entity_ids[:limit]]


def _entity_payload(
    kg: KnowledgeGraph,
    entity_id: str,
    *,
    selected: bool = False,
) -> dict[str, Any]:
    entity = kg.entities[entity_id]
    return {
        "id": entity_id,
        "label": entity.label,
        "entity_type": entity.entity_type,
        "description": entity.description,
        "aliases": list(entity.aliases),
        "source_chunks": list(entity.source_chunks),
        "degree": int(kg.graph.degree[entity_id]),
        "in_degree": int(kg.graph.in_degree(entity_id)),
        "out_degree": int(kg.graph.out_degree(entity_id)),
        "pagerank": kg.pagerank.get(entity_id, 0.0),
        "selected": selected,
    }


def _relation_payload(
    kg: KnowledgeGraph,
    relation: Relation,
    index: int,
) -> dict[str, Any]:
    return {
        "id": f"r{index}",
        "subject_id": relation.subject_id,
        "subject_label": kg.entities[relation.subject_id].label,
        "predicate": relation.predicate,
        "object_id": relation.object_id,
        "object_label": kg.entities[relation.object_id].label,
        "confidence": relation.confidence,
        "source_chunk": relation.source_chunk,
    }


def _retrieved_entity_payload(
    result: Any,
    kg: KnowledgeGraph | None = None,
) -> dict[str, Any]:
    payload = {
        "rank": result.rank,
        "entity_id": result.entity_id,
        "label": result.entity.label,
        "entity_type": result.entity.entity_type,
        "description": result.entity.description,
        "aliases": list(result.entity.aliases),
        "source_chunks": list(result.entity.source_chunks),
        "score": result.score,
        "retrieval_mode": result.retrieval_mode,
        "component_scores": dict(result.component_scores),
    }
    if kg is not None and result.entity_id in kg.entities:
        payload.update(
            {
                "degree": int(kg.graph.degree[result.entity_id]),
                "in_degree": int(kg.graph.in_degree(result.entity_id)),
                "out_degree": int(kg.graph.out_degree(result.entity_id)),
                "pagerank": kg.pagerank.get(result.entity_id, 0.0),
            }
        )
    return payload


def _context_relation_payloads(
    kg: KnowledgeGraph,
    retrieved: list[Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    retrieved_ids = {result.entity_id for result in retrieved}
    if not retrieved_ids:
        return []
    relations = [
        relation
        for relation in kg.relations
        if relation.subject_id in retrieved_ids or relation.object_id in retrieved_ids
    ]
    relations.sort(
        key=lambda relation: (
            not (
                relation.subject_id in retrieved_ids
                and relation.object_id in retrieved_ids
            ),
            -relation.confidence,
            _relation_sort_key(kg, relation),
        )
    )
    return [
        _relation_payload(kg, relation, index)
        for index, relation in enumerate(relations[:limit])
    ]


def _source_chunks_from_context(
    retrieved: list[Any],
    relation_payloads: list[dict[str, Any]],
) -> list[str]:
    chunks: list[str] = []
    for result in retrieved:
        chunks.extend(result.entity.source_chunks)
    chunks.extend(
        str(relation["source_chunk"])
        for relation in relation_payloads
        if relation.get("source_chunk")
    )
    unique_chunks: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk and chunk not in seen:
            unique_chunks.append(chunk)
            seen.add(chunk)
    return unique_chunks[:20]


def _matches_entity(entity: Any, normalized_query: str) -> bool:
    if not normalized_query:
        return True
    haystack = " ".join(
        [
            entity.label,
            entity.entity_type,
            entity.description,
            " ".join(entity.aliases),
        ]
    ).casefold()
    return normalized_query in haystack


def _relation_sort_key(
    kg: KnowledgeGraph,
    relation: Relation,
) -> tuple[str, str, str, str, str]:
    return (
        kg.entities[relation.subject_id].label.casefold(),
        relation.predicate.casefold(),
        kg.entities[relation.object_id].label.casefold(),
        relation.subject_id,
        relation.object_id,
    )


def _documents_from_payload(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw_documents = payload.get("documents", [])
    if not isinstance(raw_documents, list):
        msg = "documents must be a list."
        raise ValueError(msg)
    documents: list[dict[str, str]] = []
    for index, raw_document in enumerate(raw_documents, start=1):
        if not isinstance(raw_document, dict):
            msg = f"document {index} must be an object."
            raise ValueError(msg)
        filename = str(raw_document.get("filename", f"document_{index}.md"))
        content = str(raw_document.get("content", ""))
        documents.append({"filename": filename, "content": content})
    return documents


def _read_json_artifact(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_markdown_filename(filename: str, index: int) -> str:
    name = Path(filename).name.strip() or f"document_{index}.md"
    if not name.lower().endswith(".md"):
        name = f"{name}.md"
    return name


def _graph_path_from_params(
    params: dict[str, list[str]],
    state: WebAppState,
) -> Path:
    return resolve_project_path(
        _query_param(params, "graph"),
        default=state.default_graph_path,
    )


def _query_param(
    params: dict[str, list[str]],
    name: str,
    default: str = "",
) -> str:
    values = params.get(name)
    if not values:
        return default
    return values[0]


def _int_query_param(
    params: dict[str, list[str]],
    name: str,
    default: int,
) -> int:
    return _safe_int(_query_param(params, name, str(default)), default=default)


def _safe_int(value: object, *, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _clamp(value: int, *, minimum: int, maximum: int) -> int:
    return min(max(value, minimum), maximum)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]
