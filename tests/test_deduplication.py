from hirag_ontology.pipeline.deduplication import (
    DeduplicationAgent,
    semantic_similarity,
    token_sort_ratio,
)
from hirag_ontology.pipeline.knowledge_graph import (
    Entity,
    KnowledgeGraph,
    entity_id_from_label,
)


def test_token_sort_ratio_handles_word_order_variants() -> None:
    similarity = token_sort_ratio(
        "acute lymphoblastic leukemia",
        "lymphoblastic acute leukemia",
    )

    assert similarity == 1.0


def test_semantic_similarity_uses_cosine_when_embeddings_exist() -> None:
    assert semantic_similarity([1.0, 0.0], [1.0, 0.0], fallback=0.0) == 1.0
    assert semantic_similarity(None, [1.0, 0.0], fallback=0.42) == 0.42


def test_deduplicate_merges_punctuation_variants_and_preserves_aliases() -> None:
    kg = KnowledgeGraph()
    first_id = kg.add_entity(
        Entity(
            label="BCR-ABL inhibitor",
            entity_type="Drug",
            aliases=["TKI"],
            source_chunks=["chunk-1"],
        )
    )
    second_id = kg.add_entity(
        Entity(
            label="BCR ABL inhibitor",
            entity_type="Drug",
            aliases=["tyrosine kinase inhibitor"],
            source_chunks=["chunk-2"],
        )
    )

    result = DeduplicationAgent(threshold=0.85).deduplicate(kg)

    canonical_id = result.canonical_map[first_id]
    assert result.canonical_map[second_id] == canonical_id
    assert result.merged_count == 1
    assert len(kg.entities) == 1
    assert kg.entities[canonical_id].aliases == [
        "TKI",
        "BCR ABL inhibitor",
        "tyrosine kinase inhibitor",
    ]
    assert kg.entities[canonical_id].source_chunks == ["chunk-1", "chunk-2"]


def test_deduplicate_merges_word_order_variants() -> None:
    kg = KnowledgeGraph()
    first_id = kg.add_entity(Entity(label="acute lymphoblastic leukemia"))
    second_id = kg.add_entity(Entity(label="lymphoblastic acute leukemia"))

    result = DeduplicationAgent(threshold=0.85).deduplicate(kg)

    assert result.merged_count == 1
    assert result.canonical_map[first_id] == result.canonical_map[second_id]
    assert len(kg.entities) == 1


def test_deduplicate_does_not_merge_below_threshold() -> None:
    kg = KnowledgeGraph()
    kg.add_entity(Entity(label="imatinib"))
    kg.add_entity(Entity(label="dasatinib"))

    result = DeduplicationAgent(threshold=0.95).deduplicate(kg)

    assert result.merged_count == 0
    assert len(kg.entities) == 2


def test_deduplicate_redirects_relations_to_highest_degree_canonical() -> None:
    kg = KnowledgeGraph()
    canonical_id = kg.add_entity(Entity(label="BCR-ABL inhibitor", entity_type="Drug"))
    duplicate_id = kg.add_entity(Entity(label="BCR ABL inhibitor", entity_type="Drug"))
    condition_id = kg.add_entity(Entity(label="Ph+ ALL", entity_type="Condition"))
    symptom_id = kg.add_entity(Entity(label="nausea", entity_type="Symptom"))
    kg.add_relation_by_ids(canonical_id, "treats", condition_id)
    kg.add_relation_by_ids(canonical_id, "causes", symptom_id)
    kg.add_relation_by_ids(duplicate_id, "treats", condition_id)

    result = DeduplicationAgent(threshold=0.85).deduplicate(kg)

    assert result.canonical_map[duplicate_id] == canonical_id
    assert duplicate_id not in kg.entities
    assert all(relation.subject_id == canonical_id for relation in kg.relations)
    assert kg.graph.has_edge(canonical_id, condition_id)
    assert kg.graph.has_edge(canonical_id, symptom_id)
    assert not kg.graph.has_node(duplicate_id)


def test_deduplicate_removes_self_loops_created_by_merge() -> None:
    kg = KnowledgeGraph()
    first_id = kg.add_entity(Entity(label="BCR-ABL inhibitor"))
    second_id = kg.add_entity(Entity(label="BCR ABL inhibitor"))
    kg.add_relation_by_ids(first_id, "related_to", second_id)

    DeduplicationAgent(threshold=0.85).deduplicate(kg)

    assert kg.relations == []
    assert not list(kg.graph.edges)


def test_deduplicate_can_use_semantic_similarity_to_merge() -> None:
    kg = KnowledgeGraph()
    first_id = kg.add_entity(Entity(label="imatinib", embedding=[1.0, 0.0]))
    second_id = kg.add_entity(Entity(label="imatinib therapy", embedding=[1.0, 0.0]))

    result = DeduplicationAgent(alpha=0.6, threshold=0.85).deduplicate(kg)

    assert result.canonical_map[first_id] == result.canonical_map[second_id]
    assert len(kg.entities) == 1


def test_deduplicate_keeps_entity_ids_deterministic() -> None:
    kg = KnowledgeGraph()
    first_id = kg.add_entity(Entity(label="BCR-ABL inhibitor"))
    kg.add_entity(Entity(label="BCR ABL inhibitor"))

    DeduplicationAgent(threshold=0.85).deduplicate(kg)

    assert set(kg.entities) == {first_id}
    assert first_id == entity_id_from_label("BCR-ABL inhibitor")
