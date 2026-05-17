"""Optional Neo4j graph storage backend."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from hashlib import md5
from importlib import import_module
from typing import Any

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

_ENTITY_QUERY = """
MERGE (e:Entity {id: $id})
SET e.label = $label,
    e.entity_type = $entity_type,
    e.description = $description,
    e.aliases = $aliases,
    e.source_chunks = $source_chunks,
    e.embedding = $embedding,
    e.metadata_json = $metadata_json,
    e.pagerank = $pagerank
"""

_RELATION_QUERY_TEMPLATE = """
MATCH (s:Entity {id: $subject_id})
MATCH (o:Entity {id: $object_id})
MERGE (s)-[r:__RELATIONSHIP_TYPE__ {
    predicate: $predicate,
    relation_key: $relation_key
}]->(o)
SET r.confidence = $confidence,
    r.source_chunk = $source_chunk,
    r.metadata_json = $metadata_json
"""

_READ_ENTITIES_QUERY = """
MATCH (e:Entity)
RETURN e.id AS id,
       e.label AS label,
       e.entity_type AS entity_type,
       e.description AS description,
       e.aliases AS aliases,
       e.source_chunks AS source_chunks,
       e.embedding AS embedding,
       e.metadata_json AS metadata_json,
       e.pagerank AS pagerank
ORDER BY e.id
"""

_READ_RELATIONS_QUERY = """
MATCH (s:Entity)-[r]->(o:Entity)
RETURN s.id AS subject_id,
       r.predicate AS predicate,
       o.id AS object_id,
       r.confidence AS confidence,
       r.source_chunk AS source_chunk,
       r.metadata_json AS metadata_json
ORDER BY subject_id, predicate, object_id
"""


class Neo4jGraphStore:
    """GraphStore implementation using an optional Neo4j driver."""

    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
        driver: Any | None = None,
    ) -> None:
        self.uri = uri
        self.user = user
        self.database = database or None
        self.driver = driver or _build_driver(uri=uri, user=user, password=password)

    def close(self) -> None:
        """Close the underlying driver when supported."""
        close = getattr(self.driver, "close", None)
        if callable(close):
            close()

    def ensure_schema(self) -> None:
        """Create the entity ID uniqueness constraint when supported."""
        with self._session() as session:
            session.run(
                "CREATE CONSTRAINT hirag_entity_id IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            )

    def read_graph(self) -> KnowledgeGraph:
        """Read the entire Neo4j graph into the in-memory KnowledgeGraph."""
        kg = KnowledgeGraph()
        with self._session() as session:
            for record in session.run(_READ_ENTITIES_QUERY):
                entity_id = str(_record_get(record, "id"))
                kg.entities[entity_id] = Entity(
                    label=str(_record_get(record, "label") or ""),
                    entity_type=str(_record_get(record, "entity_type") or "Other"),
                    description=str(_record_get(record, "description") or ""),
                    aliases=_string_list(_record_get(record, "aliases")),
                    source_chunks=_string_list(_record_get(record, "source_chunks")),
                    embedding=_float_list_or_none(_record_get(record, "embedding")),
                    metadata=_decode_metadata(_record_get(record, "metadata_json")),
                )
                kg._sync_node(entity_id)
                pagerank = _record_get(record, "pagerank")
                if pagerank is not None:
                    kg.pagerank[entity_id] = float(pagerank)

            for record in session.run(_READ_RELATIONS_QUERY):
                subject_id = str(_record_get(record, "subject_id"))
                object_id = str(_record_get(record, "object_id"))
                if subject_id == object_id:
                    continue
                relation = Relation(
                    subject_id=subject_id,
                    predicate=str(_record_get(record, "predicate") or "related_to"),
                    object_id=object_id,
                    confidence=float(_record_get(record, "confidence") or 1.0),
                    source_chunk=_optional_str(_record_get(record, "source_chunk")),
                    metadata=_decode_metadata(_record_get(record, "metadata_json")),
                )
                kg.relations.append(relation)
                kg.graph.add_edge(
                    relation.subject_id,
                    relation.object_id,
                    predicate=relation.predicate,
                    confidence=relation.confidence,
                    source_chunk=relation.source_chunk,
                )
        return kg

    def write_graph(self, kg: KnowledgeGraph, *, clear: bool = False) -> None:
        """Write a KnowledgeGraph into Neo4j."""
        self.ensure_schema()
        with self._session() as session:
            if clear:
                session.run("MATCH (n) DETACH DELETE n")

            for entity_id, entity in sorted(kg.entities.items()):
                session.run(
                    _ENTITY_QUERY,
                    id=entity_id,
                    label=entity.label,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    aliases=list(entity.aliases),
                    source_chunks=list(entity.source_chunks),
                    embedding=entity.embedding,
                    metadata_json=_encode_metadata(entity.metadata),
                    pagerank=kg.pagerank.get(entity_id),
                )

            for relation in kg.relations:
                if relation.subject_id == relation.object_id:
                    continue
                relationship_type = relationship_type_from_predicate(
                    relation.predicate
                )
                session.run(
                    _RELATION_QUERY_TEMPLATE.replace(
                        "__RELATIONSHIP_TYPE__",
                        relationship_type,
                    ),
                    subject_id=relation.subject_id,
                    predicate=relation.predicate,
                    relation_key=_relation_key(relation),
                    object_id=relation.object_id,
                    confidence=relation.confidence,
                    source_chunk=relation.source_chunk,
                    metadata_json=_encode_metadata(relation.metadata),
                )

    def stats(self) -> GraphStats:
        """Return entity and relation counts from Neo4j."""
        with self._session() as session:
            entity_record = session.run(
                "MATCH (e:Entity) RETURN count(e) AS count"
            ).single()
            relation_record = session.run(
                "MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS count"
            ).single()
        return GraphStats(
            entity_count=int(_record_get(entity_record, "count") or 0),
            relation_count=int(_record_get(relation_record, "count") or 0),
        )

    def search_entities(
        self,
        query: str = "",
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[GraphNode]:
        """Search entities in Neo4j."""
        safe_limit = normalize_limit(limit, default=20, maximum=200)
        safe_offset = normalize_offset(offset)
        records = self._run_records(
            """
            MATCH (e:Entity)
            WHERE $search = ""
               OR toLower(coalesce(e.label, "")) CONTAINS $search
               OR toLower(coalesce(e.entity_type, "")) CONTAINS $search
               OR toLower(coalesce(e.description, "")) CONTAINS $search
               OR any(alias IN coalesce(e.aliases, [])
                      WHERE toLower(alias) CONTAINS $search)
            RETURN e.id AS id,
                   e.label AS label,
                   e.entity_type AS entity_type,
                   e.description AS description,
                   e.aliases AS aliases,
                   e.source_chunks AS source_chunks,
                   e.metadata_json AS metadata_json
            ORDER BY toLower(e.label), e.id
            SKIP $offset
            LIMIT $limit
            """,
            search=query.casefold().strip(),
            offset=safe_offset,
            limit=safe_limit,
        )
        return [_node_from_record(record) for record in records]

    def get_subgraph(
        self,
        entity_id: str,
        *,
        depth: int = 1,
        limit: int = 50,
        offset: int = 0,
    ) -> GraphSubgraph:
        """Return a limited local Neo4j subgraph around one entity."""
        safe_depth = normalize_depth(depth)
        safe_limit = normalize_limit(limit, default=50, maximum=200)
        safe_offset = normalize_offset(offset)
        total = self._subgraph_total(entity_id=entity_id, depth=safe_depth)
        records = self._subgraph_records(
            entity_id=entity_id,
            depth=safe_depth,
            limit=safe_limit,
            offset=safe_offset,
        )
        nodes_by_id: dict[str, GraphNode] = {}
        relations: list[GraphRelation] = []
        for record in records:
            subject = _node_from_record(record, prefix="subject")
            obj = _node_from_record(record, prefix="object")
            nodes_by_id[subject.id] = subject
            nodes_by_id[obj.id] = obj
            relations.append(
                GraphRelation(
                    subject_id=subject.id,
                    predicate=str(_record_get(record, "predicate") or "related_to"),
                    object_id=obj.id,
                    confidence=float(_record_get(record, "confidence") or 1.0),
                    source_chunk=_optional_str(_record_get(record, "source_chunk")),
                    metadata=_decode_metadata(_record_get(record, "metadata_json")),
                )
            )
        return GraphSubgraph(
            nodes=sorted(
                nodes_by_id.values(),
                key=lambda node: (node.label.casefold(), node.id),
            ),
            relations=relations,
            total_relations=total,
            limit=safe_limit,
            offset=safe_offset,
            has_more=safe_offset + safe_limit < total,
        )

    def _subgraph_total(self, *, entity_id: str, depth: int) -> int:
        query = f"""
        MATCH (center:Entity {{id: $entity_id}})
        MATCH path = (center)-[*1..{depth}]-(neighbor:Entity)
        UNWIND relationships(path) AS r
        WITH DISTINCT r
        RETURN count(r) AS count
        """
        record = self._run_single(query, entity_id=entity_id)
        return int(_record_get(record, "count") or 0)

    def _subgraph_records(
        self,
        *,
        entity_id: str,
        depth: int,
        limit: int,
        offset: int,
    ) -> list[Any]:
        query = f"""
        MATCH (center:Entity {{id: $entity_id}})
        MATCH path = (center)-[*1..{depth}]-(neighbor:Entity)
        UNWIND relationships(path) AS r
        WITH DISTINCT r
        MATCH (s:Entity)-[r]->(o:Entity)
        RETURN s.id AS subject_id,
               s.label AS subject_label,
               s.entity_type AS subject_entity_type,
               s.description AS subject_description,
               s.aliases AS subject_aliases,
               s.source_chunks AS subject_source_chunks,
               s.metadata_json AS subject_metadata_json,
               o.id AS object_id,
               o.label AS object_label,
               o.entity_type AS object_entity_type,
               o.description AS object_description,
               o.aliases AS object_aliases,
               o.source_chunks AS object_source_chunks,
               o.metadata_json AS object_metadata_json,
               r.predicate AS predicate,
               r.confidence AS confidence,
               r.source_chunk AS source_chunk,
               r.metadata_json AS metadata_json
        ORDER BY predicate, subject_label, object_label, subject_id, object_id
        SKIP $offset
        LIMIT $limit
        """
        return self._run_records(
            query,
            entity_id=entity_id,
            offset=offset,
            limit=limit,
        )

    def _run_records(self, cypher: str, **parameters: Any) -> list[Any]:
        with self._session() as session:
            return list(session.run(cypher, **parameters))

    def _run_single(self, cypher: str, **parameters: Any) -> Any:
        with self._session() as session:
            return session.run(cypher, **parameters).single()

    def _session(self) -> Any:
        if self.database:
            return self.driver.session(database=self.database)
        return self.driver.session()


def relationship_type_from_predicate(predicate: str) -> str:
    """Convert an ontology predicate to a safe Neo4j relationship type."""
    relationship_type = re.sub(r"\W+", "_", predicate.upper()).strip("_")
    if not relationship_type:
        return "RELATED_TO"
    if not relationship_type[0].isalpha():
        relationship_type = f"REL_{relationship_type}"
    return relationship_type


def _relation_key(relation: Relation) -> str:
    payload = {
        "subject_id": relation.subject_id,
        "predicate": relation.predicate,
        "object_id": relation.object_id,
        "source_chunk": relation.source_chunk,
        "metadata": relation.metadata,
    }
    raw_value = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return md5(raw_value.encode("utf-8")).hexdigest()


def _build_driver(*, uri: str, user: str, password: str) -> Any:
    try:
        neo4j_module = import_module("neo4j")
    except ImportError as error:  # pragma: no cover - optional dependency path
        msg = (
            "Neo4j support requires the optional 'neo4j' package. "
            "Install it with: uv sync --extra neo4j"
        )
        raise RuntimeError(msg) from error
    return neo4j_module.GraphDatabase.driver(uri, auth=(user, password))


def _record_get(record: Any, key: str) -> Any:
    if record is None:
        return None
    if isinstance(record, Mapping):
        return record.get(key)
    return record[key]


def _node_from_record(record: Any, *, prefix: str | None = None) -> GraphNode:
    key = _prefixed_key(prefix)
    return GraphNode(
        id=str(_record_get(record, key("id"))),
        label=str(_record_get(record, key("label")) or ""),
        entity_type=str(_record_get(record, key("entity_type")) or "Other"),
        description=str(_record_get(record, key("description")) or ""),
        aliases=_string_list(_record_get(record, key("aliases"))),
        source_chunks=_string_list(_record_get(record, key("source_chunks"))),
        metadata=_decode_metadata(_record_get(record, key("metadata_json"))),
    )


def _prefixed_key(prefix: str | None) -> Callable[[str], str]:
    if prefix is None:
        return lambda key: key
    return lambda key: f"{prefix}_{key}"


def _encode_metadata(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


def _decode_metadata(raw_value: Any) -> dict[str, Any]:
    if not raw_value:
        return {}
    if isinstance(raw_value, dict):
        return dict(raw_value)
    try:
        decoded = json.loads(str(raw_value))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _string_list(raw_value: Any) -> list[str]:
    if not raw_value:
        return []
    return [str(item) for item in raw_value]


def _float_list_or_none(raw_value: Any) -> list[float] | None:
    if raw_value is None:
        return None
    return [float(item) for item in raw_value]


def _optional_str(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    return str(raw_value)
