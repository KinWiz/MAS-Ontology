import json

import pytest

from hirag_ontology.pipeline.knowledge_graph import (
    Entity,
    KnowledgeGraph,
    entity_id_from_label,
    normalize_label,
)


def test_entity_id_uses_normalized_label() -> None:
    assert normalize_label("  Imatinib   400 mg/day ") == "imatinib 400 mg/day"
    assert entity_id_from_label("Imatinib") == entity_id_from_label(" imatinib ")


def test_add_entity_merges_existing_entity_data() -> None:
    kg = KnowledgeGraph()

    first_id = kg.add_entity(
        Entity(
            label="Imatinib",
            entity_type="Other",
            aliases=["STI-571"],
            source_chunks=["chunk-1"],
        )
    )
    second_id = kg.add_entity(
        Entity(
            label=" imatinib ",
            entity_type="Drug",
            description="BCR-ABL tyrosine kinase inhibitor",
            aliases=["Glivec", "STI-571"],
            source_chunks=["chunk-2", "chunk-1"],
            metadata={"source": "test"},
        )
    )

    entity = kg.get_entity(first_id)

    assert first_id == second_id
    assert entity.entity_type == "Drug"
    assert entity.description == "BCR-ABL tyrosine kinase inhibitor"
    assert entity.aliases == ["STI-571", "Glivec"]
    assert entity.source_chunks == ["chunk-1", "chunk-2"]
    assert entity.metadata == {"source": "test"}
    assert kg.graph.nodes[first_id]["entity_type"] == "Drug"


def test_add_relation_creates_missing_entities_and_rejects_self_loop() -> None:
    kg = KnowledgeGraph()

    kg.add_relation(
        "imatinib",
        "treats",
        "Ph+ acute lymphoblastic leukemia",
        confidence=0.9,
        source_chunk="chunk-1",
    )

    subject_id = entity_id_from_label("imatinib")
    object_id = entity_id_from_label("Ph+ acute lymphoblastic leukemia")

    assert len(kg.entities) == 2
    assert len(kg.relations) == 1
    assert kg.graph.has_edge(subject_id, object_id)
    assert kg.graph.edges[subject_id, object_id]["predicate"] == "treats"
    assert kg.relations[0].confidence == 0.9
    assert kg.relations[0].source_chunk == "chunk-1"

    with pytest.raises(ValueError, match="Self-loop"):
        kg.add_relation("imatinib", "related_to", " imatinib ")


def test_json_roundtrip_preserves_graph_data(tmp_path) -> None:
    kg = KnowledgeGraph()
    imatinib_id = kg.add_entity(
        Entity(
            label="imatinib",
            entity_type="Drug",
            aliases=["Glivec"],
            source_chunks=["chunk-1"],
        )
    )
    condition_id = kg.add_entity(
        Entity(
            label="Ph+ acute lymphoblastic leukemia",
            entity_type="Condition",
            source_chunks=["chunk-1"],
        )
    )
    kg.add_relation_by_ids(
        imatinib_id,
        "treats",
        condition_id,
        confidence=0.95,
        source_chunk="chunk-1",
    )
    kg.compute_pagerank()

    path = tmp_path / "graph.json"
    kg.save(path)

    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    loaded = KnowledgeGraph.load(path)

    assert sorted(raw_payload) == ["entities", "pagerank", "relations"]
    assert loaded.entities == kg.entities
    assert loaded.relations == kg.relations
    assert loaded.pagerank == kg.pagerank
    assert loaded.graph.has_edge(imatinib_id, condition_id)
    assert loaded.graph.edges[imatinib_id, condition_id]["confidence"] == 0.95


def test_compute_pagerank_returns_scores_for_all_entities() -> None:
    kg = KnowledgeGraph()
    kg.add_relation("imatinib", "treats", "Ph+ acute lymphoblastic leukemia")
    kg.add_relation("dasatinib", "treats", "Ph+ acute lymphoblastic leukemia")

    pagerank = kg.compute_pagerank()

    assert set(pagerank) == set(kg.entities)
    assert pytest.approx(sum(pagerank.values())) == 1.0
    assert (
        pagerank[entity_id_from_label("Ph+ acute lymphoblastic leukemia")]
        > pagerank[entity_id_from_label("imatinib")]
    )
