"""JSON-backed graph storage."""

from __future__ import annotations

from pathlib import Path

from hirag_ontology.pipeline.knowledge_graph import Entity, KnowledgeGraph, Relation
from hirag_ontology.storage.base import (
    GraphNode,
    GraphRelation,
    GraphStats,
    GraphSubgraph,
    normalize_depth,
    normalize_limit,
    normalize_offset,
)


class JsonGraphStore:
    """GraphStore implementation backed by the existing JSON format."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read_graph(self) -> KnowledgeGraph:
        """Read a graph from JSON."""
        return KnowledgeGraph.load(self.path)

    def write_graph(self, kg: KnowledgeGraph, *, clear: bool = False) -> None:
        """Write a graph to JSON."""
        del clear
        kg.save(self.path)

    def stats(self) -> GraphStats:
        """Return entity and relation counts."""
        kg = self.read_graph()
        return GraphStats(
            entity_count=len(kg.entities),
            relation_count=len(kg.relations),
        )

    def search_entities(
        self,
        query: str = "",
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[GraphNode]:
        """Search entities in the JSON graph."""
        kg = self.read_graph()
        normalized_query = query.casefold().strip()
        safe_limit = normalize_limit(limit, default=20, maximum=200)
        safe_offset = normalize_offset(offset)
        nodes = [
            _node_from_entity(entity_id, entity)
            for entity_id, entity in kg.entities.items()
            if _matches_entity(entity, normalized_query)
        ]
        nodes.sort(key=lambda node: (node.label.casefold(), node.id))
        return nodes[safe_offset : safe_offset + safe_limit]

    def get_subgraph(
        self,
        entity_id: str,
        *,
        depth: int = 1,
        limit: int = 50,
        offset: int = 0,
    ) -> GraphSubgraph:
        """Return a limited local subgraph around one entity."""
        kg = self.read_graph()
        safe_depth = normalize_depth(depth)
        safe_limit = normalize_limit(limit, default=50, maximum=200)
        safe_offset = normalize_offset(offset)

        if entity_id not in kg.entities:
            return GraphSubgraph(
                nodes=[],
                relations=[],
                total_relations=0,
                limit=safe_limit,
                offset=safe_offset,
                has_more=False,
            )

        context_ids = {entity_id, *kg.neighbors(entity_id, depth=safe_depth)}
        candidate_relations = [
            relation
            for relation in kg.relations
            if relation.subject_id in context_ids and relation.object_id in context_ids
        ]
        candidate_relations.sort(key=lambda relation: _relation_sort_key(kg, relation))
        total_relations = len(candidate_relations)
        page = candidate_relations[safe_offset : safe_offset + safe_limit]
        page_node_ids = {entity_id}
        for relation in page:
            page_node_ids.add(relation.subject_id)
            page_node_ids.add(relation.object_id)

        nodes = [
            _node_from_entity(node_id, kg.entities[node_id])
            for node_id in page_node_ids
            if node_id in kg.entities
        ]
        nodes.sort(key=lambda node: (node.label.casefold(), node.id))
        relations = [_relation_to_view(relation) for relation in page]
        return GraphSubgraph(
            nodes=nodes,
            relations=relations,
            total_relations=total_relations,
            limit=safe_limit,
            offset=safe_offset,
            has_more=safe_offset + safe_limit < total_relations,
        )


def _node_from_entity(entity_id: str, entity: Entity) -> GraphNode:
    return GraphNode(
        id=entity_id,
        label=entity.label,
        entity_type=entity.entity_type,
        description=entity.description,
        aliases=list(entity.aliases),
        source_chunks=list(entity.source_chunks),
        metadata=dict(entity.metadata),
    )


def _relation_to_view(relation: Relation) -> GraphRelation:
    return GraphRelation(
        subject_id=relation.subject_id,
        predicate=relation.predicate,
        object_id=relation.object_id,
        confidence=relation.confidence,
        source_chunk=relation.source_chunk,
        metadata=dict(relation.metadata),
    )


def _matches_entity(entity: Entity, normalized_query: str) -> bool:
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
