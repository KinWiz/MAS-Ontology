"""Extraction agent for converting text chunks into graph candidates."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hirag_ontology.llm import JsonResponse, LLMClient, LLMResponseError
from hirag_ontology.ontology import Ontology, load_ontology
from hirag_ontology.pipeline.chunking import TextChunk
from hirag_ontology.pipeline.knowledge_graph import Entity, KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class ExtractedRelation:
    """A relation extracted with endpoint labels before graph ID resolution."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source_chunk: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Structured extraction result for one chunk."""

    entities: list[Entity]
    relations: list[ExtractedRelation]
    raw: dict[str, Any] = field(default_factory=dict)


class ExtractionAgent:
    """Extract entities and relations from a text chunk using an LLM client."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        ontology: Ontology | None = None,
        cache_dir: str | Path = ".cache/hirag_ontology/extraction",
    ) -> None:
        self.llm_client = llm_client
        self.ontology = ontology or load_ontology()
        self.cache_dir = Path(cache_dir)

    def extract(self, chunk: TextChunk | str) -> ExtractionResult:
        """Extract structured entities and relations from a chunk."""
        text, chunk_id = self._chunk_text_and_id(chunk)
        cache_path = self._cache_path(text)

        if cache_path.exists():
            logger.info("Extraction cache hit: %s", cache_path.name)
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            logger.info("Extraction cache miss: %s", cache_path.name)
            prompt = self.build_prompt(text)
            payload = self._coerce_json_object(
                self.llm_client.complete_json(prompt, schema_name="extract")
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return self.parse_payload(payload, source_chunk=chunk_id)

    def extract_to_graph(
        self,
        kg: KnowledgeGraph,
        chunk: TextChunk | str,
    ) -> ExtractionResult:
        """Extract a chunk and insert extracted items into a knowledge graph."""
        result = self.extract(chunk)
        for entity in result.entities:
            kg.add_entity(entity)
        for relation in result.relations:
            kg.add_relation(
                relation.subject,
                relation.predicate,
                relation.object,
                confidence=relation.confidence,
                source_chunk=relation.source_chunk,
            )
        return result

    def build_prompt(self, text: str) -> str:
        """Build the explicit extraction prompt sent to the LLM."""
        entity_types = ", ".join(sorted(self.ontology.class_names))
        relation_types = ", ".join(sorted(self.ontology.relation_names))
        return (
            "You are a medical knowledge extraction system.\n"
            f"Allowed entity types: {entity_types}.\n"
            f"Allowed relation types: {relation_types}.\n"
            "Return only valid JSON with keys 'entities' and 'relations'.\n"
            "Each entity must include label, type, and optional description.\n"
            "Each relation must include subject, predicate, object, and optional "
            "confidence.\n"
            "Chunk text:\n"
            f"{text}"
        )

    def parse_payload(
        self,
        payload: dict[str, Any],
        *,
        source_chunk: str | None,
    ) -> ExtractionResult:
        """Parse and validate the LLM JSON object."""
        entities_payload = payload.get("entities", [])
        relations_payload = payload.get("relations", [])
        if not isinstance(entities_payload, list):
            msg = "Extraction response field 'entities' must be a list."
            raise LLMResponseError(msg)
        if not isinstance(relations_payload, list):
            msg = "Extraction response field 'relations' must be a list."
            raise LLMResponseError(msg)

        entities = [
            self._parse_entity(entity_payload, source_chunk=source_chunk)
            for entity_payload in entities_payload
        ]
        relations = [
            self._parse_relation(relation_payload, source_chunk=source_chunk)
            for relation_payload in relations_payload
        ]
        return ExtractionResult(entities=entities, relations=relations, raw=payload)

    def _parse_entity(
        self,
        payload: Any,
        *,
        source_chunk: str | None,
    ) -> Entity:
        if not isinstance(payload, dict):
            msg = "Each extracted entity must be an object."
            raise LLMResponseError(msg)
        label = str(payload.get("label", "")).strip()
        if not label:
            msg = "Each extracted entity must include a non-empty label."
            raise LLMResponseError(msg)

        aliases_payload = payload.get("aliases", [])
        if aliases_payload is None:
            aliases_payload = []
        if not isinstance(aliases_payload, list):
            msg = "Entity aliases must be a list when provided."
            raise LLMResponseError(msg)

        source_chunks = [source_chunk] if source_chunk is not None else []
        return Entity(
            label=label,
            entity_type=str(payload.get("type", "Other") or "Other"),
            description=str(payload.get("description", "") or ""),
            aliases=[str(alias) for alias in aliases_payload],
            source_chunks=source_chunks,
        )

    def _parse_relation(
        self,
        payload: Any,
        *,
        source_chunk: str | None,
    ) -> ExtractedRelation:
        if not isinstance(payload, dict):
            msg = "Each extracted relation must be an object."
            raise LLMResponseError(msg)

        subject = str(payload.get("subject", "")).strip()
        predicate = str(payload.get("predicate", "")).strip()
        obj = str(payload.get("object", "")).strip()
        if not subject or not predicate or not obj:
            msg = "Each relation must include subject, predicate, and object."
            raise LLMResponseError(msg)

        try:
            confidence = float(payload.get("confidence", 1.0))
        except (TypeError, ValueError) as error:
            msg = "Relation confidence must be numeric."
            raise LLMResponseError(msg) from error

        return ExtractedRelation(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source_chunk=source_chunk,
        )

    def _cache_path(self, text: str) -> Path:
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{text_hash}.json"

    @staticmethod
    def _coerce_json_object(response: JsonResponse) -> dict[str, Any]:
        if isinstance(response, dict):
            return response
        if isinstance(response, str):
            try:
                payload = json.loads(response)
            except json.JSONDecodeError as error:
                msg = "Malformed JSON returned by LLM."
                raise LLMResponseError(msg) from error
            if isinstance(payload, dict):
                return payload

        msg = "LLM JSON response must be an object."
        raise LLMResponseError(msg)

    @staticmethod
    def _chunk_text_and_id(chunk: TextChunk | str) -> tuple[str, str | None]:
        if isinstance(chunk, TextChunk):
            return chunk.text, chunk.chunk_id
        return chunk, None


def extraction_result_to_dict(result: ExtractionResult) -> dict[str, Any]:
    """Convert an extraction result to JSON-compatible data."""
    return {
        "entities": [asdict(entity) for entity in result.entities],
        "relations": [asdict(relation) for relation in result.relations],
        "raw": result.raw,
    }
