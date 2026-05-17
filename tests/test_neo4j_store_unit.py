from typing import Any

import pytest

from hirag_ontology.pipeline.knowledge_graph import Entity, KnowledgeGraph
from hirag_ontology.storage.neo4j_store import (
    Neo4jGraphStore,
    relationship_type_from_predicate,
)


class FakeResult:
    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.records = records or []

    def __iter__(self):
        return iter(self.records)

    def single(self):
        return self.records[0] if self.records else None


class FakeSession:
    def __init__(self, driver: "FakeDriver") -> None:
        self.driver = driver

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def run(self, query: str, **parameters: Any) -> FakeResult:
        self.driver.calls.append((query, parameters))
        compact_query = " ".join(query.split())
        if "count(e) AS count" in compact_query:
            return FakeResult([{"count": 2}])
        if "count(r) AS count" in compact_query:
            return FakeResult([{"count": 1}])
        if "RETURN e.id AS id" in compact_query:
            return FakeResult(
                [
                    {
                        "id": "drug-1",
                        "label": "imatinib",
                        "entity_type": "Drug",
                        "description": "BCR-ABL inhibitor",
                        "aliases": ["Glivec"],
                        "source_chunks": ["chunk-1"],
                        "embedding": None,
                        "metadata_json": '{"source": "unit"}',
                        "pagerank": 0.4,
                    },
                    {
                        "id": "condition-1",
                        "label": "Ph-positive ALL",
                        "entity_type": "Condition",
                        "description": "",
                        "aliases": ["Ph+ ALL"],
                        "source_chunks": [],
                        "embedding": None,
                        "metadata_json": "{}",
                        "pagerank": 0.6,
                    },
                ]
            )
        if "RETURN s.id AS subject_id" in compact_query:
            return FakeResult(
                [
                    {
                        "subject_id": "drug-1",
                        "subject_label": "imatinib",
                        "subject_entity_type": "Drug",
                        "subject_description": "BCR-ABL inhibitor",
                        "subject_aliases": ["Glivec"],
                        "subject_source_chunks": ["chunk-1"],
                        "subject_metadata_json": '{"source": "unit"}',
                        "predicate": "treats",
                        "object_id": "condition-1",
                        "object_label": "Ph-positive ALL",
                        "object_entity_type": "Condition",
                        "object_description": "",
                        "object_aliases": ["Ph+ ALL"],
                        "object_source_chunks": [],
                        "object_metadata_json": "{}",
                        "confidence": 0.9,
                        "source_chunk": "chunk-1",
                        "metadata_json": "{}",
                    }
                ]
            )
        return FakeResult()


class FakeDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def session(self, **kwargs: Any) -> FakeSession:
        self.calls.append(("SESSION", kwargs))
        return FakeSession(self)

    def close(self) -> None:
        self.closed = True


def _graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    drug_id = kg.add_entity(
        Entity(label="imatinib", entity_type="Drug", metadata={"source": "unit"})
    )
    condition_id = kg.add_entity(
        Entity(label="Ph-positive ALL", entity_type="Condition")
    )
    kg.add_relation_by_ids(
        drug_id,
        "treats",
        condition_id,
        confidence=0.9,
        source_chunk="chunk-1",
    )
    kg.compute_pagerank()
    return kg


def test_relationship_type_from_predicate_sanitizes_values() -> None:
    assert relationship_type_from_predicate("diagnosed_by") == "DIAGNOSED_BY"
    assert relationship_type_from_predicate("123 invalid!") == "REL_123_INVALID"
    assert relationship_type_from_predicate("") == "RELATED_TO"


def test_neo4j_store_writes_graph_with_clear_and_schema() -> None:
    driver = FakeDriver()
    store = Neo4jGraphStore(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="secret",
        driver=driver,
    )

    store.write_graph(_graph(), clear=True)

    queries = [" ".join(query.split()) for query, _ in driver.calls]
    assert any("CREATE CONSTRAINT hirag_entity_id" in query for query in queries)
    assert any("MATCH (n) DETACH DELETE n" in query for query in queries)
    assert any("MERGE (e:Entity {id: $id})" in query for query in queries)
    assert any("MERGE (s)-[r:TREATS" in query for query in queries)


def test_neo4j_store_reads_graph_and_stats_from_driver() -> None:
    driver = FakeDriver()
    store = Neo4jGraphStore(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="secret",
        driver=driver,
    )

    kg = store.read_graph()
    stats = store.stats()
    store.close()

    assert len(kg.entities) == 2
    assert len(kg.relations) == 1
    assert kg.pagerank == {"drug-1": 0.4, "condition-1": 0.6}
    assert stats.entity_count == 2
    assert stats.relation_count == 1
    assert driver.closed is True


def test_neo4j_store_searches_entities_from_driver() -> None:
    driver = FakeDriver()
    store = Neo4jGraphStore(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="secret",
        driver=driver,
    )

    results = store.search_entities("all", limit=2)

    assert [entity.label for entity in results] == ["imatinib", "Ph-positive ALL"]
    assert any(call[1].get("search") == "all" for call in driver.calls)


def test_neo4j_store_returns_limited_subgraph_from_driver() -> None:
    driver = FakeDriver()
    store = Neo4jGraphStore(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="secret",
        driver=driver,
    )

    subgraph = store.get_subgraph("drug-1", limit=1)

    assert [node.id for node in subgraph.nodes] == ["drug-1", "condition-1"]
    assert len(subgraph.relations) == 1
    assert subgraph.total_relations == 1
    assert subgraph.has_more is False


def test_neo4j_store_requires_optional_package_when_no_driver(monkeypatch) -> None:
    def fail_import(*args: object, **kwargs: object) -> None:
        raise ImportError("missing")

    monkeypatch.setattr("builtins.__import__", fail_import)

    with pytest.raises(RuntimeError, match="optional 'neo4j' package"):
        Neo4jGraphStore(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="secret",
        )
