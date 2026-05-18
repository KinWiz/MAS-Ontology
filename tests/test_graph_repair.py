from hirag_ontology.ontology import load_ontology
from hirag_ontology.pipeline.graph_repair import GraphRepairOptions, repair_graph
from hirag_ontology.pipeline.knowledge_graph import Entity, KnowledgeGraph
from hirag_ontology.pipeline.validator import ValidationAgent


def test_repair_graph_reverses_relaxes_and_validates() -> None:
    kg = KnowledgeGraph()
    condition_id = kg.add_entity(Entity(label="Ph+ ALL", entity_type="Condition"))
    lab_id = kg.add_entity(Entity(label="FISH", entity_type="LabTest"))
    protocol_id = kg.add_entity(Entity(label="ALL protocol", entity_type="Procedure"))
    dose_id = kg.add_entity(Entity(label="600 mg daily", entity_type="Other"))
    drug_id = kg.add_entity(Entity(label="imatinib", entity_type="Drug"))

    kg.add_relation_by_ids(lab_id, "diagnosed_by", condition_id)
    kg.add_relation_by_ids(protocol_id, "treats", condition_id)
    kg.add_relation_by_ids(drug_id, "dosage_is", dose_id)
    kg.add_relation_by_ids(drug_id, "unknown_predicate", condition_id)

    ontology = load_ontology()
    report = repair_graph(kg, ontology=ontology)
    validation = ValidationAgent(ontology).validate(kg)

    assert validation["violations"] == []
    assert report["validation_before"]["violation_count"] > 0
    assert report["validation_after"]["violation_count"] == 0
    assert report["actions"]["reversed"] == 1
    assert report["actions"]["domain_range_relaxed"] == 2
    assert report["actions"]["invalid_predicate_relaxed"] == 1
    assert report["actions"]["inferred_entity_types"] == 0
    assert any(
        relation.subject_id == condition_id
        and relation.predicate == "diagnosed_by"
        and relation.object_id == lab_id
        for relation in kg.relations
    )
    assert any(
        relation.subject_id == protocol_id
        and relation.predicate == "related_to"
        and relation.object_id == condition_id
        for relation in kg.relations
    )


def test_repair_graph_can_infer_other_type_with_enough_votes() -> None:
    kg = KnowledgeGraph()
    drug_a = kg.add_entity(Entity(label="imatinib", entity_type="Drug"))
    drug_b = kg.add_entity(Entity(label="dasatinib", entity_type="Drug"))
    dose_id = kg.add_entity(Entity(label="600 mg daily", entity_type="Other"))

    kg.add_relation_by_ids(drug_a, "dosage_is", dose_id)
    kg.add_relation_by_ids(drug_b, "dosage_is", dose_id)

    report = repair_graph(
        kg,
        ontology=load_ontology(),
        options=GraphRepairOptions(infer_other_types=True),
    )

    assert kg.entities[dose_id].entity_type == "DosageRegimen"
    assert report["actions"]["inferred_entity_types"] == 1
