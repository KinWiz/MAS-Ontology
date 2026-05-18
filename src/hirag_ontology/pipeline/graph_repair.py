"""Validation-aware graph repair utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from hirag_ontology.ontology import Ontology, load_ontology
from hirag_ontology.pipeline.knowledge_graph import KnowledgeGraph, Relation
from hirag_ontology.pipeline.validator import ValidationAgent, ValidationResult


@dataclass(frozen=True)
class GraphRepairOptions:
    """Options controlling conservative graph repair."""

    infer_other_types: bool = False
    min_type_votes: int = 2
    min_type_vote_share: float = 0.67
    relaxed_confidence: float = 0.45
    deduplicate_relations: bool = True


def repair_graph(
    kg: KnowledgeGraph,
    *,
    ontology: Ontology | None = None,
    options: GraphRepairOptions | None = None,
) -> dict[str, Any]:
    """Repair a graph so saved relations satisfy the ontology constraints.

    The repair is intentionally conservative: it reverses relations only when
    the ontology unambiguously supports the reverse direction. Remaining
    domain/range conflicts are retained as weaker ``related_to`` edges instead
    of being deleted.
    """
    repair_options = options or GraphRepairOptions()
    active_ontology = ontology or load_ontology()
    validator = ValidationAgent(active_ontology)
    validation_before = validator.validate(kg)
    predicate_distribution_before = Counter(
        relation.predicate for relation in kg.relations
    )

    action_counts: Counter[str] = Counter()
    inferred_types: list[dict[str, Any]] = []
    action_counts["inferred_entity_types"] = 0
    if repair_options.infer_other_types:
        inferred_types = _infer_other_entity_types(
            kg,
            active_ontology,
            repair_options,
        )
        action_counts["inferred_entity_types"] = len(inferred_types)

    repaired_relations: list[Relation] = []
    for relation in kg.relations:
        repaired = _repair_relation(
            relation,
            kg,
            active_ontology,
            repair_options,
        )
        if repaired is None:
            action_counts["removed_self_loops"] += 1
            continue
        repaired_relations.append(repaired.relation)
        action_counts[repaired.action] += 1

    duplicate_removed = 0
    if repair_options.deduplicate_relations:
        repaired_relations, duplicate_removed = _deduplicate_relations(
            repaired_relations
        )
    action_counts["deduplicated_relations"] = duplicate_removed

    kg.relations = repaired_relations
    _rebuild_edges(kg)
    kg.compute_pagerank()

    validation_after = validator.validate(kg)
    predicate_distribution_after = Counter(
        relation.predicate for relation in kg.relations
    )
    return {
        "validation_before": _validation_summary(validation_before),
        "validation_after": _validation_summary(validation_after),
        "relation_count_before": sum(predicate_distribution_before.values()),
        "relation_count_after": len(kg.relations),
        "entity_count": len(kg.entities),
        "actions": dict(sorted(action_counts.items())),
        "inferred_types": inferred_types[:100],
        "inferred_type_count": len(inferred_types),
        "predicate_distribution_before": dict(
            sorted(predicate_distribution_before.items())
        ),
        "predicate_distribution_after": dict(
            sorted(predicate_distribution_after.items())
        ),
    }


@dataclass(frozen=True)
class _RelationRepair:
    relation: Relation
    action: str


def _repair_relation(
    relation: Relation,
    kg: KnowledgeGraph,
    ontology: Ontology,
    options: GraphRepairOptions,
) -> _RelationRepair | None:
    if relation.subject_id == relation.object_id:
        return None

    relation_spec = ontology.relations.get(relation.predicate)
    if relation_spec is None:
        return _RelationRepair(
            _relaxed_relation(
                relation,
                options,
                action="invalid_predicate_relaxed",
            ),
            "invalid_predicate_relaxed",
        )

    subject = kg.entities[relation.subject_id]
    obj = kg.entities[relation.object_id]
    if _types_match(subject.entity_type, relation_spec.domain) and _types_match(
        obj.entity_type,
        relation_spec.range,
    ):
        return _RelationRepair(relation, "kept")

    if _types_match(obj.entity_type, relation_spec.domain) and _types_match(
        subject.entity_type,
        relation_spec.range,
    ):
        return _RelationRepair(_reversed_relation(relation), "reversed")

    return _RelationRepair(
        _relaxed_relation(
            relation,
            options,
            action="domain_range_relaxed",
        ),
        "domain_range_relaxed",
    )


def _infer_other_entity_types(
    kg: KnowledgeGraph,
    ontology: Ontology,
    options: GraphRepairOptions,
) -> list[dict[str, Any]]:
    votes: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for relation in kg.relations:
        relation_spec = ontology.relations.get(relation.predicate)
        if relation_spec is None or relation.predicate == "related_to":
            continue
        subject = kg.entities.get(relation.subject_id)
        obj = kg.entities.get(relation.object_id)
        if (
            subject is not None
            and subject.entity_type == "Other"
            and relation_spec.domain != "Other"
        ):
            votes[relation.subject_id][relation_spec.domain] += 1
        if (
            obj is not None
            and obj.entity_type == "Other"
            and relation_spec.range != "Other"
        ):
            votes[relation.object_id][relation_spec.range] += 1

    changes: list[dict[str, Any]] = []
    for entity_id, entity_votes in votes.items():
        top_type, top_votes = entity_votes.most_common(1)[0]
        total_votes = sum(entity_votes.values())
        if top_votes < options.min_type_votes:
            continue
        if top_votes / max(total_votes, 1) < options.min_type_vote_share:
            continue
        entity = kg.entities[entity_id]
        previous_type = entity.entity_type
        entity.entity_type = top_type
        entity.metadata["repair_previous_type"] = previous_type
        entity.metadata["repair_type_votes"] = dict(entity_votes)
        if entity_id in kg.graph:
            kg.graph.nodes[entity_id]["entity_type"] = top_type
        changes.append(
            {
                "entity_id": entity_id,
                "label": entity.label,
                "from": previous_type,
                "to": top_type,
                "votes": dict(entity_votes),
            }
        )
    return changes


def _types_match(actual: str, expected: str) -> bool:
    return expected == "Other" or actual == expected


def _reversed_relation(relation: Relation) -> Relation:
    metadata = dict(relation.metadata)
    metadata["repair_action"] = "reversed"
    metadata["repair_original_subject_id"] = relation.subject_id
    metadata["repair_original_object_id"] = relation.object_id
    return Relation(
        subject_id=relation.object_id,
        predicate=relation.predicate,
        object_id=relation.subject_id,
        confidence=relation.confidence,
        source_chunk=relation.source_chunk,
        metadata=metadata,
    )


def _relaxed_relation(
    relation: Relation,
    options: GraphRepairOptions,
    *,
    action: str,
) -> Relation:
    metadata = dict(relation.metadata)
    metadata["repair_action"] = action
    metadata["repair_original_predicate"] = relation.predicate
    return Relation(
        subject_id=relation.subject_id,
        predicate="related_to",
        object_id=relation.object_id,
        confidence=min(relation.confidence, options.relaxed_confidence),
        source_chunk=relation.source_chunk,
        metadata=metadata,
    )


def _deduplicate_relations(
    relations: list[Relation],
) -> tuple[list[Relation], int]:
    by_key: dict[tuple[str, str, str, str | None], Relation] = {}
    ordered_keys: list[tuple[str, str, str, str | None]] = []
    removed = 0
    for relation in relations:
        key = (
            relation.subject_id,
            relation.predicate,
            relation.object_id,
            relation.source_chunk,
        )
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = relation
            ordered_keys.append(key)
            continue
        removed += 1
        if relation.confidence > existing.confidence:
            by_key[key] = relation
    return [by_key[key] for key in ordered_keys], removed


def _rebuild_edges(kg: KnowledgeGraph) -> None:
    kg.graph.clear_edges()
    for relation in kg.relations:
        kg.graph.add_edge(
            relation.subject_id,
            relation.object_id,
            predicate=relation.predicate,
            confidence=relation.confidence,
            source_chunk=relation.source_chunk,
        )


def _validation_summary(result: ValidationResult) -> dict[str, Any]:
    violation_types = Counter(str(item["type"]) for item in result["violations"])
    return {
        "consistency_score": result["consistency_score"],
        "violation_count": len(result["violations"]),
        "counts": dict(result["counts"]),
        "violation_types": dict(sorted(violation_types.items())),
    }
