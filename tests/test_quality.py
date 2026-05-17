from pytest import approx

from hirag_ontology.ontology import load_ontology
from hirag_ontology.pipeline.knowledge_graph import Entity, KnowledgeGraph
from hirag_ontology.pipeline.quality import (
    QualityWeights,
    compute_coverage,
    compute_precision,
    compute_quality,
    compute_redundancy,
)


def test_quality_components_follow_spec_approximations() -> None:
    ontology = load_ontology()
    kg = KnowledgeGraph()
    drug_id = kg.add_entity(
        Entity(
            label="imatinib",
            entity_type="Drug",
            aliases=["Glivec", "BCR-ABL inhibitor"],
        )
    )
    condition_id = kg.add_entity(
        Entity(
            label="Ph+ acute lymphoblastic leukemia",
            entity_type="Condition",
        )
    )
    kg.add_entity(Entity(label="untyped note", entity_type="Other"))
    kg.add_relation_by_ids(drug_id, "treats", condition_id, confidence=0.5)
    kg.add_relation_by_ids(drug_id, "related_to", condition_id, confidence=1.0)

    assert compute_coverage(kg, ontology) == approx(2 / 8)
    assert compute_precision(kg) == approx(0.75)
    assert compute_redundancy(kg) == approx(2 / 5)


def test_compute_quality_uses_default_weights_and_validation_result() -> None:
    kg = KnowledgeGraph()
    drug_id = kg.add_entity(Entity(label="imatinib", entity_type="Drug"))
    condition_id = kg.add_entity(
        Entity(
            label="Ph+ acute lymphoblastic leukemia",
            entity_type="Condition",
            aliases=["Ph+ ALL"],
        )
    )
    kg.add_relation_by_ids(drug_id, "treats", condition_id, confidence=0.8)

    scores = compute_quality(
        kg,
        validation_result={"consistency_score": 0.9},
    )

    expected_q = (0.3 * (2 / 8)) + (0.3 * 0.9) + (0.2 * 0.8) - (0.2 * (1 / 3))
    assert scores.coverage == approx(2 / 8)
    assert scores.consistency == approx(0.9)
    assert scores.precision == approx(0.8)
    assert scores.redundancy == approx(1 / 3)
    assert scores.q == approx(expected_q)


def test_compute_quality_accepts_custom_weights_and_default_precision() -> None:
    kg = KnowledgeGraph()

    scores = compute_quality(
        kg,
        validation_result={"consistency_score": 1.0},
        weights=QualityWeights(
            coverage=0.25,
            consistency=0.25,
            precision=0.25,
            redundancy=0.25,
        ),
        default_precision=0.4,
    )

    assert scores.coverage == 0.0
    assert scores.consistency == 1.0
    assert scores.precision == 0.4
    assert scores.redundancy == 0.0
    assert scores.q == approx(0.35)
