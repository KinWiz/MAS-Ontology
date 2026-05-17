"""Simple graph reasoning rules for the MVP pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

from hirag_ontology.pipeline.knowledge_graph import KnowledgeGraph, Relation


@dataclass
class ReasoningAgent:
    """Infer missing relations from graph structure."""

    confidence: float = 0.7

    def apply(self, kg: KnowledgeGraph) -> dict[str, Any]:
        """Apply MVP reasoning rules and mutate the graph with inferred edges."""
        inferred_relations = self.infer_shared_condition_relations(kg)
        for relation in inferred_relations:
            kg.relations.append(relation)
            kg.graph.add_edge(
                relation.subject_id,
                relation.object_id,
                predicate=relation.predicate,
                confidence=relation.confidence,
                source_chunk=relation.source_chunk,
                metadata=relation.metadata,
            )

        return {
            "rules_applied": ["shared_condition_rule"],
            "suggested_relations": len(inferred_relations),
            "added_relations": len(inferred_relations),
            "suggestions": [
                self._suggestion_payload(kg, relation)
                for relation in inferred_relations
            ],
        }

    def infer_shared_condition_relations(
        self,
        kg: KnowledgeGraph,
    ) -> list[Relation]:
        """Infer related_to between drugs that treat the same condition."""
        drugs_by_condition = self._drugs_by_condition(kg)
        existing = {
            (relation.subject_id, relation.predicate, relation.object_id)
            for relation in kg.relations
        }
        inferred: list[Relation] = []

        for condition_id, drug_ids in sorted(
            drugs_by_condition.items(),
            key=lambda item: kg.entities[item[0]].label.casefold(),
        ):
            ordered_drugs = sorted(
                drug_ids,
                key=lambda entity_id: (
                    kg.entities[entity_id].label.casefold(),
                    entity_id,
                ),
            )
            for subject_id, object_id in combinations(ordered_drugs, 2):
                if subject_id == object_id:
                    continue

                relation_key = (subject_id, "related_to", object_id)
                reverse_key = (object_id, "related_to", subject_id)
                if relation_key in existing or reverse_key in existing:
                    continue

                relation = Relation(
                    subject_id=subject_id,
                    predicate="related_to",
                    object_id=object_id,
                    confidence=self.confidence,
                    metadata={
                        "inferred_by": "shared_condition_rule",
                        "condition_id": condition_id,
                    },
                )
                inferred.append(relation)
                existing.add(relation_key)

        return inferred

    @staticmethod
    def _drugs_by_condition(kg: KnowledgeGraph) -> dict[str, set[str]]:
        drugs_by_condition: dict[str, set[str]] = {}
        for relation in kg.relations:
            if relation.predicate != "treats":
                continue

            subject = kg.entities.get(relation.subject_id)
            obj = kg.entities.get(relation.object_id)
            if subject is None or obj is None:
                continue
            if subject.entity_type != "Drug" or obj.entity_type != "Condition":
                continue

            drugs_by_condition.setdefault(relation.object_id, set()).add(
                relation.subject_id,
            )
        return drugs_by_condition

    @staticmethod
    def _suggestion_payload(
        kg: KnowledgeGraph,
        relation: Relation,
    ) -> dict[str, str]:
        condition_id = str(relation.metadata.get("condition_id", ""))
        return {
            "subject": kg.entities[relation.subject_id].label,
            "predicate": relation.predicate,
            "object": kg.entities[relation.object_id].label,
            "condition": kg.entities[condition_id].label if condition_id else "",
        }


def apply_reasoning(kg: KnowledgeGraph) -> dict[str, Any]:
    """Convenience wrapper for one-shot reasoning."""
    return ReasoningAgent().apply(kg)
