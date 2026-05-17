from hirag_ontology.pipeline.knowledge_graph import Entity, KnowledgeGraph
from hirag_ontology.storage import JsonGraphStore


def _graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    drug_id = kg.add_entity(
        Entity(
            label="imatinib",
            entity_type="Drug",
            description="BCR-ABL tyrosine kinase inhibitor",
            aliases=["Glivec"],
        )
    )
    condition_id = kg.add_entity(
        Entity(
            label="Ph-positive ALL",
            entity_type="Condition",
            aliases=["Ph+ ALL"],
        )
    )
    test_id = kg.add_entity(Entity(label="FISH", entity_type="LabTest"))
    kg.add_relation_by_ids(drug_id, "treats", condition_id, confidence=0.9)
    kg.add_relation_by_ids(condition_id, "diagnosed_by", test_id, confidence=0.8)
    kg.compute_pagerank()
    return kg


def test_json_store_roundtrips_existing_graph_format(tmp_path) -> None:
    path = tmp_path / "graph.json"
    store = JsonGraphStore(path)
    kg = _graph()

    store.write_graph(kg)
    loaded = store.read_graph()

    assert loaded.entities == kg.entities
    assert loaded.relations == kg.relations
    assert loaded.pagerank == kg.pagerank
    assert store.stats().entity_count == 3
    assert store.stats().relation_count == 2


def test_json_store_entity_search_is_limited_and_deterministic(tmp_path) -> None:
    path = tmp_path / "graph.json"
    JsonGraphStore(path).write_graph(_graph())
    store = JsonGraphStore(path)

    results = store.search_entities("ALL", limit=1)

    assert len(results) == 1
    assert results[0].label == "Ph-positive ALL"


def test_json_store_subgraph_is_paginated(tmp_path) -> None:
    path = tmp_path / "graph.json"
    kg = _graph()
    JsonGraphStore(path).write_graph(kg)
    condition_id = next(
        entity_id
        for entity_id, entity in kg.entities.items()
        if entity.label == "Ph-positive ALL"
    )

    subgraph = JsonGraphStore(path).get_subgraph(condition_id, limit=1)

    assert len(subgraph.relations) == 1
    assert subgraph.total_relations == 2
    assert subgraph.has_more is True
    assert any(node.label == "Ph-positive ALL" for node in subgraph.nodes)
