"""Graph validation against the MVP ontology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hirag_ontology.ontology import Ontology, load_ontology
from hirag_ontology.pipeline.knowledge_graph import (
    KnowledgeGraph,
    Relation,
)

Violation = dict[str, Any]
ValidationResult = dict[str, Any]


@dataclass
class ValidationAgent:
    """Validate and lightly repair a knowledge graph."""

    ontology: Ontology

    def validate(self, kg: KnowledgeGraph) -> ValidationResult:
        """Check entity types, predicates, domain/range constraints, self-loops."""
        violations: list[Violation] = []
        counts = {
            "valid_entity_type": 0,
            "valid_predicate": 0,
            "domain_constraint": 0,
            "range_constraint": 0,
            "no_self_loops": 0,
        }

        for entity_id, entity in kg.entities.items():
            if entity.entity_type not in self.ontology.class_names:
                counts["valid_entity_type"] += 1
                violations.append(
                    {
                        "type": "invalid_entity_type",
                        "entity_id": entity_id,
                        "label": entity.label,
                        "actual": entity.entity_type,
                        "expected": sorted(self.ontology.class_names),
                    }
                )

        for relation_index, relation in enumerate(kg.relations):
            subject = kg.entities.get(relation.subject_id)
            obj = kg.entities.get(relation.object_id)

            if relation.subject_id == relation.object_id:
                counts["no_self_loops"] += 1
                violations.append(
                    self._relation_violation(
                        "self_loop",
                        relation_index,
                        relation,
                    )
                )

            relation_spec = self.ontology.relations.get(relation.predicate)
            if relation_spec is None:
                counts["valid_predicate"] += 1
                violations.append(
                    self._relation_violation(
                        "invalid_predicate",
                        relation_index,
                        relation,
                        expected=sorted(self.ontology.relation_names),
                        actual=relation.predicate,
                    )
                )
                continue

            if (
                subject is not None
                and relation_spec.domain != "Other"
                and subject.entity_type != relation_spec.domain
            ):
                counts["domain_constraint"] += 1
                violations.append(
                    self._relation_violation(
                        "domain_violation",
                        relation_index,
                        relation,
                        expected=relation_spec.domain,
                        actual=subject.entity_type,
                    )
                )

            if (
                obj is not None
                and relation_spec.range != "Other"
                and obj.entity_type != relation_spec.range
            ):
                counts["range_constraint"] += 1
                violations.append(
                    self._relation_violation(
                        "range_violation",
                        relation_index,
                        relation,
                        expected=relation_spec.range,
                        actual=obj.entity_type,
                    )
                )

        total_checks = max(len(kg.entities) + (4 * len(kg.relations)), 1)
        consistency_score = max(0.0, 1.0 - (len(violations) / total_checks))

        return {
            "consistency_score": consistency_score,
            "violations": violations,
            "counts": counts,
        }

    def auto_repair(
        self,
        kg: KnowledgeGraph,
        validation_result: ValidationResult | None = None,
    ) -> dict[str, Any]:
        """Convert unknown entity types to Other and remove self-loop relations."""
        result = (
            validation_result
            if validation_result is not None
            else self.validate(kg)
        )
        repaired_unknown_types = 0
        removed_self_loops = 0

        for entity_id, entity in kg.entities.items():
            if entity.entity_type not in self.ontology.class_names:
                entity.entity_type = "Other"
                if entity_id in kg.graph:
                    kg.graph.nodes[entity_id]["entity_type"] = "Other"
                repaired_unknown_types += 1

        repaired_relations: list[Relation] = []
        for relation in kg.relations:
            if relation.subject_id == relation.object_id:
                removed_self_loops += 1
                continue
            repaired_relations.append(relation)

        if removed_self_loops:
            kg.relations = repaired_relations
            self._rebuild_edges(kg)

        repaired_result = self.validate(kg)
        return {
            "repaired_unknown_types": repaired_unknown_types,
            "removed_self_loops": removed_self_loops,
            "violations_before": len(result["violations"]),
            "violations_after": len(repaired_result["violations"]),
        }

    @staticmethod
    def _relation_violation(
        violation_type: str,
        relation_index: int,
        relation: Relation,
        *,
        expected: Any | None = None,
        actual: Any | None = None,
    ) -> Violation:
        violation: Violation = {
            "type": violation_type,
            "relation_index": relation_index,
            "predicate": relation.predicate,
            "subject_id": relation.subject_id,
            "object_id": relation.object_id,
        }
        if expected is not None:
            violation["expected"] = expected
        if actual is not None:
            violation["actual"] = actual
        return violation

    @staticmethod
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


def validate_graph(
    kg: KnowledgeGraph,
    ontology: Ontology | None = None,
) -> ValidationResult:
    """Convenience wrapper for one-shot graph validation."""
    return ValidationAgent(ontology or load_ontology()).validate(kg)


def auto_repair(
    kg: KnowledgeGraph,
    ontology: Ontology | None = None,
    validation_result: ValidationResult | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for one-shot graph repair."""
    return ValidationAgent(ontology or load_ontology()).auto_repair(
        kg,
        validation_result,
    )
