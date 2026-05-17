"""Hybrid entity retrieval over semantic, lexical, and structural signals."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from hirag_ontology.pipeline.knowledge_graph import Entity, KnowledgeGraph
from hirag_ontology.retrieval.rrf import rrf_fusion


class EmbeddingProvider(Protocol):
    """Minimal embedding provider interface used by the retriever."""

    def encode(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
    ) -> list[list[float]]:
        """Encode texts into embedding vectors."""


class FakeEmbeddingProvider:
    """Deterministic embedding provider for tests."""

    def __init__(
        self,
        vectors: dict[str, list[float]] | None = None,
        *,
        dimensions: int = 3,
    ) -> None:
        self.vectors = {
            key.casefold(): list(value)
            for key, value in (vectors or {}).items()
        }
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def encode(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
    ) -> list[list[float]]:
        """Encode using configured fixtures or stable hash fallback vectors."""
        self.calls.append(list(texts))
        vectors = [self._vector_for_text(text) for text in texts]
        if normalize:
            return [_normalize_vector(vector) for vector in vectors]
        return vectors

    def _vector_for_text(self, text: str) -> list[float]:
        folded = text.casefold()
        if folded in self.vectors:
            return list(self.vectors[folded])

        for key, vector in sorted(self.vectors.items(), key=lambda item: -len(item[0])):
            if key in folded:
                return list(vector)

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [
            (digest[index] / 255.0)
            for index in range(self.dimensions)
        ]


class RetrievalMode(StrEnum):
    """Supported retrieval modes."""

    SEMANTIC_ONLY = "semantic_only"
    LEXICAL_ONLY = "lexical_only"
    STRUCTURAL_ONLY = "structural_only"
    HYBRID_RRF = "hybrid_rrf"


@dataclass
class RetrievedEntity:
    """One retrieved entity with score details."""

    entity_id: str
    entity: Entity
    score: float
    rank: int
    retrieval_mode: str
    component_scores: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    """Retrieve entities from a knowledge graph using multiple signals."""

    def __init__(
        self,
        kg: KnowledgeGraph,
        embedding_provider: EmbeddingProvider,
        mode: RetrievalMode = RetrievalMode.HYBRID_RRF,
        rrf_k: int = 60,
    ) -> None:
        self.kg = kg
        self.embedding_provider = embedding_provider
        self.mode = mode
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedEntity]:
        """Retrieve top-K entities for a query."""
        if top_k <= 0 or not self.kg.entities:
            return []

        if self.mode == RetrievalMode.SEMANTIC_ONLY:
            ranked = self._semantic_scores(query)
            return self._to_results(ranked, top_k, self.mode.value, "semantic")
        if self.mode == RetrievalMode.LEXICAL_ONLY:
            ranked = self._lexical_scores(query)
            return self._to_results(ranked, top_k, self.mode.value, "lexical")
        if self.mode == RetrievalMode.STRUCTURAL_ONLY:
            ranked = self._structural_scores()
            return self._to_results(ranked, top_k, self.mode.value, "structural")

        semantic = self._semantic_scores(query)
        lexical = self._lexical_scores(query)
        structural = self._structural_scores()
        component_scores = {
            "semantic": dict(semantic),
            "lexical": dict(lexical),
            "structural": dict(structural),
        }
        fused = rrf_fusion(
            [
                [entity_id for entity_id, _ in semantic],
                [entity_id for entity_id, _ in lexical],
                [entity_id for entity_id, _ in structural],
            ],
            k=self.rrf_k,
        )
        return [
            RetrievedEntity(
                entity_id=entity_id,
                entity=self.kg.entities[entity_id],
                score=score,
                rank=rank,
                retrieval_mode=self.mode.value,
                component_scores={
                    name: scores.get(entity_id, 0.0)
                    for name, scores in component_scores.items()
                },
            )
            for rank, (entity_id, score) in enumerate(fused[:top_k], start=1)
        ]

    def _semantic_scores(self, query: str) -> list[tuple[str, float]]:
        entity_items = list(self.kg.entities.items())
        query_vector = self.embedding_provider.encode([query], normalize=True)[0]
        missing_items = [
            (entity_id, entity)
            for entity_id, entity in entity_items
            if entity.embedding is None
        ]
        if missing_items:
            vectors = self.embedding_provider.encode(
                [entity_document(entity) for _, entity in missing_items],
                normalize=True,
            )
            for (_, entity), vector in zip(missing_items, vectors, strict=True):
                entity.embedding = vector

        scores = [
            (
                entity_id,
                cosine_similarity(query_vector, entity.embedding or []),
            )
            for entity_id, entity in entity_items
        ]
        return self._sort_scores(scores)

    def _lexical_scores(self, query: str) -> list[tuple[str, float]]:
        entity_items = list(self.kg.entities.items())
        documents = [entity_document(entity) for _, entity in entity_items]
        tokenized_documents = [tokenize(document) for document in documents]
        scores = bm25_scores(tokenize(query), tokenized_documents)
        return self._sort_scores(
            [
                (entity_id, score)
                for (entity_id, _), score in zip(entity_items, scores, strict=True)
            ]
        )

    def _structural_scores(self) -> list[tuple[str, float]]:
        if set(self.kg.pagerank) != set(self.kg.entities):
            self.kg.compute_pagerank()
        return self._sort_scores(
            [
                (entity_id, self.kg.pagerank.get(entity_id, 0.0))
                for entity_id in self.kg.entities
            ]
        )

    def _to_results(
        self,
        ranked: list[tuple[str, float]],
        top_k: int,
        retrieval_mode: str,
        component_name: str,
    ) -> list[RetrievedEntity]:
        return [
            RetrievedEntity(
                entity_id=entity_id,
                entity=self.kg.entities[entity_id],
                score=score,
                rank=rank,
                retrieval_mode=retrieval_mode,
                component_scores={component_name: score},
            )
            for rank, (entity_id, score) in enumerate(ranked[:top_k], start=1)
        ]

    def _sort_scores(self, scores: list[tuple[str, float]]) -> list[tuple[str, float]]:
        return sorted(
            scores,
            key=lambda item: (
                -item[1],
                self.kg.entities[item[0]].label.casefold(),
                item[0],
            ),
        )


def entity_document(entity: Entity) -> str:
    """Build the text representation indexed for retrieval."""
    return " ".join(
        part
        for part in [
            entity.label,
            entity.entity_type,
            entity.description,
            " ".join(entity.aliases),
        ]
        if part
    )


def tokenize(text: str) -> list[str]:
    """Tokenize by lowercase whitespace for the MVP."""
    return text.casefold().split()


def bm25_scores(
    query_tokens: list[str],
    documents: list[list[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Compute BM25 scores for tokenized documents."""
    if not documents:
        return []
    if not query_tokens:
        return [0.0 for _ in documents]

    document_count = len(documents)
    average_length = sum(len(document) for document in documents) / document_count
    average_length = average_length or 1.0
    document_frequencies = Counter(
        token
        for document in documents
        for token in set(document)
    )

    scores: list[float] = []
    for document in documents:
        frequencies = Counter(document)
        document_length = len(document) or 1
        score = 0.0
        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if frequency == 0:
                continue
            idf = math.log(
                1.0
                + (
                    (document_count - document_frequencies[token] + 0.5)
                    / (document_frequencies[token] + 0.5)
                )
            )
            denominator = frequency + (
                k1 * (1.0 - b + (b * document_length / average_length))
            )
            score += idf * ((frequency * (k1 + 1.0)) / denominator)
        scores.append(score)
    return scores


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    return dot_product / (left_norm * right_norm)


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]
