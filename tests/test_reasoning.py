from hirag_ontology.pipeline.knowledge_graph import Entity, KnowledgeGraph
from hirag_ontology.pipeline.reasoning import ReasoningAgent


def _drug_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    condition_id = kg.add_entity(
        Entity(
            label="Ph+ acute lymphoblastic leukemia",
            entity_type="Condition",
        )
    )
    imatinib_id = kg.add_entity(Entity(label="imatinib", entity_type="Drug"))
    dasatinib_id = kg.add_entity(Entity(label="dasatinib", entity_type="Drug"))
    kg.add_relation_by_ids(imatinib_id, "treats", condition_id, confidence=0.95)
    kg.add_relation_by_ids(dasatinib_id, "treats", condition_id, confidence=0.92)
    return kg


def test_reasoning_infers_related_to_for_drugs_sharing_condition() -> None:
    kg = _drug_graph()

    stats = ReasoningAgent().apply(kg)

    inferred = [
        relation
        for relation in kg.relations
        if relation.predicate == "related_to"
    ]
    assert stats["added_relations"] == 1
    assert stats["suggested_relations"] == 1
    assert len(inferred) == 1
    assert inferred[0].confidence == 0.7
    assert inferred[0].metadata["inferred_by"] == "shared_condition_rule"
    assert stats["suggestions"] == [
        {
            "subject": "dasatinib",
            "predicate": "related_to",
            "object": "imatinib",
            "condition": "Ph+ acute lymphoblastic leukemia",
        }
    ]


def test_reasoning_does_not_add_duplicate_or_reverse_related_to() -> None:
    kg = _drug_graph()
    imatinib_id = next(
        entity_id
        for entity_id, entity in kg.entities.items()
        if entity.label == "imatinib"
    )
    dasatinib_id = next(
        entity_id
        for entity_id, entity in kg.entities.items()
        if entity.label == "dasatinib"
    )
    kg.add_relation_by_ids(imatinib_id, "related_to", dasatinib_id)

    stats = ReasoningAgent().apply(kg)

    related_relations = [
        relation
        for relation in kg.relations
        if relation.predicate == "related_to"
    ]
    assert stats["added_relations"] == 0
    assert len(related_relations) == 1


def test_reasoning_ignores_non_drug_or_non_condition_treats_edges() -> None:
    kg = KnowledgeGraph()
    drug_id = kg.add_entity(Entity(label="imatinib", entity_type="Drug"))
    procedure_id = kg.add_entity(
        Entity(label="induction therapy", entity_type="Procedure")
    )
    symptom_id = kg.add_entity(Entity(label="nausea", entity_type="Symptom"))
    kg.add_relation_by_ids(drug_id, "treats", procedure_id)
    kg.add_relation_by_ids(procedure_id, "treats", symptom_id)

    stats = ReasoningAgent().apply(kg)

    assert stats["added_relations"] == 0
    assert all(relation.predicate != "related_to" for relation in kg.relations)
