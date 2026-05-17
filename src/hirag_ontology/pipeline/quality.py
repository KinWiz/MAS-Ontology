"""Quality functional for MVP knowledge graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hirag_ontology.ontology import Ontology, load_ontology
from hirag_ontology.pipeline.knowledge_graph import KnowledgeGraph
from hirag_ontology.pipeline.validator import validate_graph


@dataclass(frozen=True)
class QualityScores:
    """Computed graph quality components."""

    coverage: float
    consistency: float
    precision: float
    redundancy: float
    q: float


@dataclass(frozen=True)
class QualityWeights:
    """Weights for the MVP quality functional."""

    coverage: float = 0.3
    consistency: float = 0.3
    precision: float = 0.2
    redundancy: float = 0.2


DEFAULT_QUALITY_WEIGHTS = QualityWeights()


def compute_quality(
    kg: KnowledgeGraph,
    *,
    ontology: Ontology | None = None,
    validation_result: dict[str, Any] | None = None,
    weights: QualityWeights = DEFAULT_QUALITY_WEIGHTS,
    default_precision: float = 1.0,
) -> QualityScores:
    """Compute the MVP graph quality functional."""
    loaded_ontology = ontology or load_ontology()
    coverage = compute_coverage(kg, loaded_ontology)
    consistency = compute_consistency(
        kg,
        ontology=loaded_ontology,
        validation_result=validation_result,
    )
    precision = compute_precision(kg, default_precision=default_precision)
    redundancy = compute_redundancy(kg)
    q = (
        weights.coverage * coverage
        + weights.consistency * consistency
        + weights.precision * precision
        - weights.redundancy * redundancy
    )
    return QualityScores(
        coverage=coverage,
        consistency=consistency,
        precision=precision,
        redundancy=redundancy,
        q=q,
    )


def compute_coverage(kg: KnowledgeGraph, ontology: Ontology) -> float:
    """Return represented non-Other ontology classes divided by all such classes."""
    target_classes = ontology.class_names - {"Other"}
    if not target_classes:
        return 0.0

    represented = {
        entity.entity_type
        for entity in kg.entities.values()
        if entity.entity_type in target_classes
    }
    return len(represented) / len(target_classes)


def compute_consistency(
    kg: KnowledgeGraph,
    *,
    ontology: Ontology,
    validation_result: dict[str, Any] | None = None,
) -> float:
    """Return validator consistency, computing it when not supplied."""
    result = (
        validation_result
        if validation_result is not None
        else validate_graph(kg, ontology)
    )
    return float(result.get("consistency_score", 0.0))


def compute_precision(
    kg: KnowledgeGraph,
    *,
    default_precision: float = 1.0,
) -> float:
    """Estimate precision as average relation confidence for the MVP."""
    if not kg.relations:
        return _clamp(default_precision)
    return _clamp(
        sum(relation.confidence for relation in kg.relations) / len(kg.relations),
    )


def compute_redundancy(kg: KnowledgeGraph) -> float:
    """Approximate redundancy by the fraction of alias mentions."""
    alias_count = sum(len(entity.aliases) for entity in kg.entities.values())
    denominator = len(kg.entities) + alias_count
    if denominator == 0:
        return 0.0
    return alias_count / denominator


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
