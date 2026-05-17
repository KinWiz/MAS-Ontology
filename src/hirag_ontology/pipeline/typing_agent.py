"""Entity typing agent backed by the LLM abstraction."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hirag_ontology.llm import JsonResponse, LLMClient, LLMResponseError
from hirag_ontology.ontology import Ontology, load_ontology
from hirag_ontology.pipeline.knowledge_graph import (
    Entity,
    KnowledgeGraph,
    normalize_label,
)

logger = logging.getLogger(__name__)


@dataclass
class TypingResult:
    """Structured result of assigning one ontology class to an entity."""

    assigned_class: str
    confidence: float
    rationale: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False


class TypingAgent:
    """Assign ontology classes to entities with deterministic caching."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        ontology: Ontology | None = None,
        cache_dir: str | Path = ".cache/hirag_ontology/typing",
    ) -> None:
        self.llm_client = llm_client
        self.ontology = ontology or load_ontology()
        self.cache_dir = Path(cache_dir)

    def type_entity(self, entity: Entity) -> TypingResult:
        """Assign one ontology class to an entity and update it in place."""
        normalized_label = normalize_label(entity.label)
        cache_path = self._cache_path(normalized_label)
        cache_hit = cache_path.exists()

        if cache_hit:
            logger.info("Typing cache hit: %s", cache_path.name)
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            logger.info("Typing cache miss: %s", cache_path.name)
            prompt = self.build_prompt(entity)
            payload = self._coerce_json_object(
                self.llm_client.complete_json(
                    prompt,
                    schema_name=f"type:{normalized_label}",
                )
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        result = self.parse_payload(payload, cache_hit=cache_hit)
        entity.entity_type = result.assigned_class
        return result

    def type_graph(self, kg: KnowledgeGraph) -> dict[str, Any]:
        """Type every entity in a graph and update NetworkX node metadata."""
        cache_hits = 0
        assigned_counts: dict[str, int] = {}

        for entity_id, entity in kg.entities.items():
            result = self.type_entity(entity)
            if result.cache_hit:
                cache_hits += 1
            assigned_counts[result.assigned_class] = (
                assigned_counts.get(result.assigned_class, 0) + 1
            )
            if entity_id in kg.graph:
                kg.graph.nodes[entity_id]["entity_type"] = result.assigned_class

        return {
            "typed_count": len(kg.entities),
            "cache_hits": cache_hits,
            "cache_misses": len(kg.entities) - cache_hits,
            "assigned_counts": assigned_counts,
        }

    def build_prompt(self, entity: Entity) -> str:
        """Build an explicit typing prompt from entity fields and ontology classes."""
        class_lines = [
            f"- {name}: {definition.description}"
            for name, definition in sorted(self.ontology.classes.items())
        ]
        class_block = "\n".join(class_lines)
        aliases = ", ".join(entity.aliases) if entity.aliases else "<none>"
        return (
            "You are a medical ontology typing agent.\n"
            "Assign exactly one ontology class to the entity.\n"
            "Return only valid JSON with keys 'class', 'confidence', and "
            "'rationale'.\n"
            "Allowed classes:\n"
            f"{class_block}\n\n"
            f"Entity label: {entity.label}\n"
            f"Description: {entity.description or '<none>'}\n"
            f"Aliases: {aliases}"
        )

    def parse_payload(
        self,
        payload: dict[str, Any],
        *,
        cache_hit: bool,
    ) -> TypingResult:
        """Parse a raw LLM response and enforce ontology fallback rules."""
        returned_class = str(payload.get("class", "Other") or "Other").strip()
        assigned_class = (
            returned_class
            if returned_class in self.ontology.class_names
            else "Other"
        )

        try:
            confidence = float(payload.get("confidence", 1.0))
        except (TypeError, ValueError) as error:
            msg = "Typing confidence must be numeric."
            raise LLMResponseError(msg) from error

        return TypingResult(
            assigned_class=assigned_class,
            confidence=confidence,
            rationale=str(payload.get("rationale", "") or ""),
            raw=payload,
            cache_hit=cache_hit,
        )

    def _cache_path(self, normalized_label: str) -> Path:
        cache_key = hashlib.md5(normalized_label.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{cache_key}.json"

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

        msg = "LLM typing response must be a JSON object."
        raise LLMResponseError(msg)
