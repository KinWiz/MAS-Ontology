"""In-memory knowledge graph model backed by NetworkX."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_python


def normalize_label(label: str) -> str:
    """Normalize an entity label while preserving medically meaningful symbols."""
    normalized = label.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
    normalized = re.sub(r"([,.;:])(?=\S)", r"\1 ", normalized)
    return normalized


def entity_id_from_label(label: str) -> str:
    """Return the deterministic MD5 entity ID for a label."""
    return hashlib.md5(normalize_label(label).encode("utf-8")).hexdigest()


def _merge_unique(existing: list[str], incoming: list[str]) -> list[str]:
    merged = list(existing)
    seen = set(existing)
    for item in incoming:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


@dataclass
class Entity:
    """A typed medical concept node."""

    label: str
    entity_type: str = "Other"
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    source_chunks: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    """A directed typed relation between two entity IDs."""

    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0
    source_chunk: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """A small JSON-serializable knowledge graph for the MVP."""

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self.graph: nx.DiGraph = nx.DiGraph()
        self.pagerank: dict[str, float] = {}

    def add_entity(self, entity: Entity) -> str:
        """Add or merge an entity and return its deterministic ID."""
        entity_id = entity_id_from_label(entity.label)

        if entity_id in self.entities:
            existing = self.entities[entity_id]
            existing.aliases = _merge_unique(existing.aliases, entity.aliases)
            existing.source_chunks = _merge_unique(
                existing.source_chunks,
                entity.source_chunks,
            )
            if existing.entity_type == "Other" and entity.entity_type != "Other":
                existing.entity_type = entity.entity_type
            if not existing.description and entity.description:
                existing.description = entity.description
            if existing.embedding is None and entity.embedding is not None:
                existing.embedding = list(entity.embedding)
            existing.metadata.update(entity.metadata)
        else:
            self.entities[entity_id] = Entity(
                label=entity.label,
                entity_type=entity.entity_type,
                description=entity.description,
                aliases=list(entity.aliases),
                source_chunks=list(entity.source_chunks),
                embedding=(
                    list(entity.embedding)
                    if entity.embedding is not None
                    else None
                ),
                metadata=dict(entity.metadata),
            )

        self._sync_node(entity_id)
        return entity_id

    def add_relation(
        self,
        subject_label: str,
        predicate: str,
        object_label: str,
        confidence: float = 1.0,
        source_chunk: str | None = None,
    ) -> None:
        """Add a relation, creating missing endpoint entities when needed."""
        subject_id = self.add_entity(Entity(label=subject_label))
        object_id = self.add_entity(Entity(label=object_label))
        self.add_relation_by_ids(
            subject_id,
            predicate,
            object_id,
            confidence=confidence,
            source_chunk=source_chunk,
        )

    def add_relation_by_ids(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        confidence: float = 1.0,
        source_chunk: str | None = None,
    ) -> None:
        """Add a relation between existing entity IDs."""
        if subject_id == object_id:
            msg = "Self-loop relations are not allowed"
            raise ValueError(msg)
        if subject_id not in self.entities:
            msg = f"Unknown subject entity ID: {subject_id}"
            raise KeyError(msg)
        if object_id not in self.entities:
            msg = f"Unknown object entity ID: {object_id}"
            raise KeyError(msg)

        relation = Relation(
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            confidence=confidence,
            source_chunk=source_chunk,
        )
        self.relations.append(relation)
        self.graph.add_edge(
            subject_id,
            object_id,
            predicate=predicate,
            confidence=confidence,
            source_chunk=source_chunk,
        )

    def compute_pagerank(
        self,
        alpha: float = 0.85,
        max_iter: int = 200,
    ) -> dict[str, float]:
        """Compute PageRank over the current graph."""
        self.pagerank = _pagerank_python(
            self.graph,
            alpha=alpha,
            max_iter=max_iter,
            weight=None,
        )
        return dict(self.pagerank)

    def save(self, path: str | Path) -> None:
        """Persist the graph to JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entities": {
                entity_id: asdict(entity)
                for entity_id, entity in sorted(self.entities.items())
            },
            "relations": [asdict(relation) for relation in self.relations],
            "pagerank": dict(sorted(self.pagerank.items())),
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a graph from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        kg = cls()

        for entity_id, entity_payload in payload.get("entities", {}).items():
            kg.entities[entity_id] = Entity(**entity_payload)
            kg._sync_node(entity_id)

        for relation_payload in payload.get("relations", []):
            relation = Relation(**relation_payload)
            if relation.subject_id == relation.object_id:
                msg = "Saved graph contains a self-loop relation"
                raise ValueError(msg)
            kg.relations.append(relation)
            kg.graph.add_edge(
                relation.subject_id,
                relation.object_id,
                predicate=relation.predicate,
                confidence=relation.confidence,
                source_chunk=relation.source_chunk,
            )

        kg.pagerank = {
            str(entity_id): float(score)
            for entity_id, score in payload.get("pagerank", {}).items()
        }
        return kg

    def neighbors(self, entity_id: str, depth: int = 1) -> list[str]:
        """Return neighbor IDs within a directed ego-network depth."""
        if entity_id not in self.graph:
            return []
        if depth < 1:
            return []

        seen = {entity_id}
        frontier = {entity_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node_id in frontier:
                next_frontier.update(self.graph.successors(node_id))
                next_frontier.update(self.graph.predecessors(node_id))
            next_frontier.difference_update(seen)
            seen.update(next_frontier)
            frontier = next_frontier

        seen.remove(entity_id)
        return sorted(seen, key=lambda item: self.entities[item].label)

    def get_entity(self, entity_id: str) -> Entity:
        """Return an entity by ID."""
        return self.entities[entity_id]

    def _sync_node(self, entity_id: str) -> None:
        entity = self.entities[entity_id]
        self.graph.add_node(
            entity_id,
            label=entity.label,
            entity_type=entity.entity_type,
            description=entity.description,
        )
