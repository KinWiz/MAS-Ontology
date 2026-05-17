"""Entity deduplication for the in-memory knowledge graph."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import combinations

from hirag_ontology.pipeline.knowledge_graph import (
    Entity,
    KnowledgeGraph,
    Relation,
    normalize_label,
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


@dataclass
class DeduplicationResult:
    """Summary of a deduplication pass."""

    entity_count_before: int
    entity_count_after: int
    merged_count: int
    clusters: list[list[str]] = field(default_factory=list)
    canonical_map: dict[str, str] = field(default_factory=dict)


class DeduplicationAgent:
    """Merge near-duplicate entity nodes in a knowledge graph."""

    def __init__(self, *, alpha: float = 0.6, threshold: float = 0.85) -> None:
        if not 0.0 <= alpha <= 1.0:
            msg = "alpha must be between 0 and 1"
            raise ValueError(msg)
        if not 0.0 <= threshold <= 1.0:
            msg = "threshold must be between 0 and 1"
            raise ValueError(msg)
        self.alpha = alpha
        self.threshold = threshold

    def deduplicate(self, kg: KnowledgeGraph) -> DeduplicationResult:
        """Cluster duplicates, merge aliases, redirect edges, and remove loops."""
        before = len(kg.entities)
        if before < 2:
            return DeduplicationResult(before, before, 0)

        union_find = _UnionFind(kg.entities.keys())
        for left_id, right_id in self._candidate_pairs(kg):
            left = kg.entities[left_id]
            right = kg.entities[right_id]
            if self.hybrid_similarity(left, right) >= self.threshold:
                union_find.union(left_id, right_id)

        raw_clusters = union_find.clusters()
        duplicate_clusters = [
            sorted(cluster, key=lambda entity_id: kg.entities[entity_id].label)
            for cluster in raw_clusters
            if len(cluster) > 1
        ]
        if not duplicate_clusters:
            return DeduplicationResult(before, before, 0)

        canonical_map = self._canonical_map(kg, raw_clusters)
        kg.entities = self._merge_entities(kg, canonical_map)
        kg.relations = self._redirect_relations(kg.relations, canonical_map)
        self._rebuild_graph(kg)
        kg.pagerank = {}

        after = len(kg.entities)
        return DeduplicationResult(
            entity_count_before=before,
            entity_count_after=after,
            merged_count=before - after,
            clusters=duplicate_clusters,
            canonical_map=canonical_map,
        )

    def hybrid_similarity(self, left: Entity, right: Entity) -> float:
        """Compute alpha-weighted semantic and lexical similarity."""
        lexical = token_sort_ratio(left.label, right.label)
        semantic = semantic_similarity(
            left.embedding,
            right.embedding,
            fallback=lexical,
        )
        return (self.alpha * semantic) + ((1.0 - self.alpha) * lexical)

    def _candidate_pairs(self, kg: KnowledgeGraph) -> set[tuple[str, str]]:
        block_index: dict[str, set[str]] = defaultdict(set)
        for entity_id, entity in kg.entities.items():
            for token in blocking_tokens(entity.label):
                block_index[token].add(entity_id)

        pairs: set[tuple[str, str]] = set()
        for entity_ids in block_index.values():
            for left_id, right_id in combinations(sorted(entity_ids), 2):
                pairs.add((left_id, right_id))
        return pairs

    def _canonical_map(
        self,
        kg: KnowledgeGraph,
        clusters: list[set[str]],
    ) -> dict[str, str]:
        canonical_map: dict[str, str] = {}
        insertion_order = {
            entity_id: index for index, entity_id in enumerate(kg.entities)
        }
        for cluster in clusters:
            canonical_id = min(
                cluster,
                key=lambda entity_id: (
                    -kg.graph.degree(entity_id),
                    insertion_order[entity_id],
                    normalize_label(kg.entities[entity_id].label),
                    entity_id,
                ),
            )
            for entity_id in cluster:
                canonical_map[entity_id] = canonical_id
        return canonical_map

    def _merge_entities(
        self,
        kg: KnowledgeGraph,
        canonical_map: dict[str, str],
    ) -> dict[str, Entity]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for entity_id in kg.entities:
            grouped[canonical_map[entity_id]].append(entity_id)

        merged_entities: dict[str, Entity] = {}
        for canonical_id, entity_ids in grouped.items():
            canonical = kg.entities[canonical_id]
            aliases = list(canonical.aliases)
            source_chunks = list(canonical.source_chunks)
            metadata = dict(canonical.metadata)
            description = canonical.description
            embedding = (
                list(canonical.embedding)
                if canonical.embedding is not None
                else None
            )

            for entity_id in sorted(entity_ids):
                entity = kg.entities[entity_id]
                if entity_id != canonical_id and entity.label not in aliases:
                    aliases.append(entity.label)
                aliases = _merge_unique(aliases, entity.aliases)
                source_chunks = _merge_unique(source_chunks, entity.source_chunks)
                metadata.update(entity.metadata)
                if not description and entity.description:
                    description = entity.description
                if embedding is None and entity.embedding is not None:
                    embedding = list(entity.embedding)

            merged_entities[canonical_id] = Entity(
                label=canonical.label,
                entity_type=canonical.entity_type,
                description=description,
                aliases=aliases,
                source_chunks=source_chunks,
                embedding=embedding,
                metadata=metadata,
            )

        return dict(sorted(merged_entities.items()))

    def _redirect_relations(
        self,
        relations: list[Relation],
        canonical_map: dict[str, str],
    ) -> list[Relation]:
        redirected: list[Relation] = []
        seen: set[tuple[str, str, str, float, str | None]] = set()
        for relation in relations:
            subject_id = canonical_map[relation.subject_id]
            object_id = canonical_map[relation.object_id]
            if subject_id == object_id:
                continue

            dedupe_key = (
                subject_id,
                relation.predicate,
                object_id,
                relation.confidence,
                relation.source_chunk,
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            redirected.append(
                Relation(
                    subject_id=subject_id,
                    predicate=relation.predicate,
                    object_id=object_id,
                    confidence=relation.confidence,
                    source_chunk=relation.source_chunk,
                    metadata=dict(relation.metadata),
                )
            )
        return redirected

    @staticmethod
    def _rebuild_graph(kg: KnowledgeGraph) -> None:
        kg.graph.clear()
        for entity_id, entity in kg.entities.items():
            kg.graph.add_node(
                entity_id,
                label=entity.label,
                entity_type=entity.entity_type,
                description=entity.description,
            )
        for relation in kg.relations:
            kg.graph.add_edge(
                relation.subject_id,
                relation.object_id,
                predicate=relation.predicate,
                confidence=relation.confidence,
                source_chunk=relation.source_chunk,
            )


def token_sort_ratio(left: str, right: str) -> float:
    """Return SequenceMatcher ratio over sorted normalized label tokens."""
    left_tokens = sorted(label_tokens(left))
    right_tokens = sorted(label_tokens(right))
    if not left_tokens or not right_tokens:
        return SequenceMatcher(
            None,
            normalize_label(left),
            normalize_label(right),
        ).ratio()
    return SequenceMatcher(
        None,
        " ".join(left_tokens),
        " ".join(right_tokens),
    ).ratio()


def semantic_similarity(
    left: list[float] | None,
    right: list[float] | None,
    *,
    fallback: float,
) -> float:
    """Return cosine similarity when embeddings exist, otherwise fallback."""
    if left is None or right is None or len(left) != len(right):
        return fallback
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return fallback
    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    return max(0.0, min(1.0, dot_product / (left_norm * right_norm)))


def label_tokens(label: str) -> list[str]:
    """Tokenize a label for lexical comparison."""
    return re.findall(r"[a-zа-яё0-9]+", normalize_label(label), flags=re.IGNORECASE)


def blocking_tokens(label: str) -> set[str]:
    """Return non-stopword tokens used for duplicate candidate blocking."""
    return {token for token in label_tokens(label) if token not in STOPWORDS}


def _merge_unique(existing: list[str], incoming: list[str]) -> list[str]:
    merged = list(existing)
    seen = set(existing)
    for item in incoming:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent: dict[str, str] = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        canonical_root = min(left_root, right_root)
        other_root = max(left_root, right_root)
        self.parent[other_root] = canonical_root

    def clusters(self) -> list[set[str]]:
        clusters: dict[str, set[str]] = defaultdict(set)
        for item in self.parent:
            clusters[self.find(item)].add(item)
        return list(clusters.values())
