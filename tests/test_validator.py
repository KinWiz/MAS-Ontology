from hirag_ontology.ontology import load_ontology
from hirag_ontology.pipeline.knowledge_graph import (
    Entity,
    KnowledgeGraph,
    Relation,
    entity_id_from_label,
)
from hirag_ontology.pipeline.validator import ValidationAgent


def _validator() -> ValidationAgent:
    return ValidationAgent(load_ontology())


def _typed_graph(subject_type: str, predicate: str, object_type: str) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    subject_id = kg.add_entity(Entity(label="imatinib", entity_type=subject_type))
    object_id = kg.add_entity(
        Entity(
            label="Ph+ acute lymphoblastic leukemia",
            entity_type=object_type,
        )
    )
    kg.add_relation_by_ids(subject_id, predicate, object_id)
    return kg


def test_default_ontology_contains_mvp_classes_and_relations() -> None:
    ontology = load_ontology()

    assert ontology.class_names == {
        "Drug",
        "Condition",
        "Procedure",
        "Symptom",
        "AnatomicalStructure",
        "DosageRegimen",
        "LabTest",
        "Organization",
        "Other",
    }
    assert ontology.relation_names == {
        "treats",
        "causes",
        "contraindicated_for",
        "part_of",
        "diagnosed_by",
        "dosage_is",
        "related_to",
    }


def test_valid_graph_has_no_violations() -> None:
    kg = _typed_graph("Drug", "treats", "Condition")

    result = _validator().validate(kg)

    assert result["violations"] == []
    assert result["consistency_score"] == 1.0


def test_invalid_predicate_is_reported() -> None:
    kg = _typed_graph("Drug", "improves", "Condition")

    result = _validator().validate(kg)

    assert result["counts"]["valid_predicate"] == 1
    assert result["violations"][0]["type"] == "invalid_predicate"
    assert result["violations"][0]["actual"] == "improves"


def test_domain_violation_is_reported() -> None:
    kg = _typed_graph("LabTest", "treats", "Condition")

    result = _validator().validate(kg)

    assert result["counts"]["domain_constraint"] == 1
    assert result["violations"][0]["type"] == "domain_violation"
    assert result["violations"][0]["expected"] == "Drug"
    assert result["violations"][0]["actual"] == "LabTest"


def test_range_violation_is_reported() -> None:
    kg = _typed_graph("Drug", "treats", "LabTest")

    result = _validator().validate(kg)

    assert result["counts"]["range_constraint"] == 1
    assert result["violations"][0]["type"] == "range_violation"
    assert result["violations"][0]["expected"] == "Condition"
    assert result["violations"][0]["actual"] == "LabTest"


def test_unknown_entity_type_is_reported_and_repaired() -> None:
    kg = _typed_graph("Medication", "treats", "Condition")
    validator = _validator()

    result = validator.validate(kg)
    repair = validator.auto_repair(kg, result)

    subject_id = entity_id_from_label("imatinib")

    assert result["counts"]["valid_entity_type"] == 1
    assert repair["repaired_unknown_types"] == 1
    assert kg.entities[subject_id].entity_type == "Other"
    assert kg.graph.nodes[subject_id]["entity_type"] == "Other"


def test_auto_repair_removes_self_loops() -> None:
    kg = KnowledgeGraph()
    entity_id = kg.add_entity(Entity(label="imatinib", entity_type="Drug"))
    kg.relations.append(Relation(entity_id, "related_to", entity_id))
    kg.graph.add_edge(entity_id, entity_id, predicate="related_to", confidence=1.0)
    validator = _validator()

    result = validator.validate(kg)
    repair = validator.auto_repair(kg, result)

    assert result["counts"]["no_self_loops"] == 1
    assert repair["removed_self_loops"] == 1
    assert kg.relations == []
    assert not kg.graph.has_edge(entity_id, entity_id)
    assert validator.validate(kg)["counts"]["no_self_loops"] == 0
