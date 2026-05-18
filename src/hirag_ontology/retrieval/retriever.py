"""Hybrid entity retrieval over semantic, lexical, and structural signals."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from hirag_ontology.config import load_embedding_settings
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


class OpenAIEmbeddingProvider:
    """OpenAI-compatible multilingual embedding provider."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        request_timeout_seconds: float = 120.0,
    ) -> None:
        if not api_key:
            msg = (
                "EMBEDDING_API_KEY or OPENAI_API_KEY is required for "
                "openai embeddings."
            )
            raise ValueError(msg)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.request_timeout_seconds = request_timeout_seconds

    def encode(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
    ) -> list[list[float]]:
        """Encode text through an OpenAI-compatible /embeddings endpoint."""
        if not texts:
            return []
        payload = self._post_json(
            f"{self.base_url}/embeddings",
            {
                "model": self.model,
                "input": texts,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        data = payload.get("data")
        if not isinstance(data, list) or not all(
            isinstance(item, dict) for item in data
        ):
            msg = "Embedding response must contain a data list."
            raise RuntimeError(msg)
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        vectors = [_embedding_from_payload(item) for item in ordered]
        if len(vectors) != len(texts):
            msg = "Embedding response length does not match input length."
            raise RuntimeError(msg)
        if normalize:
            return [_normalize_vector(vector) for vector in vectors]
        return vectors

    def _post_json(
        self,
        url: str,
        payload: dict[str, object],
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **(headers or {}),
            },
            method="POST",
        )
        try:
            with urlopen(  # noqa: S310 - user-configured local/OpenAI endpoint.
                request,
                timeout=self.request_timeout_seconds,
            ) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            msg = f"Embedding request failed with HTTP {error.code}."
            raise RuntimeError(msg) from error
        except (OSError, json.JSONDecodeError) as error:
            msg = "Embedding request failed."
            raise RuntimeError(msg) from error
        if not isinstance(decoded, dict):
            msg = "Embedding response must be a JSON object."
            raise RuntimeError(msg)
        return decoded


class OllamaEmbeddingProvider:
    """Local Ollama embedding provider."""

    def __init__(
        self,
        *,
        model: str = "mxbai-embed-large",
        base_url: str = "http://localhost:11434",
        request_timeout_seconds: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.request_timeout_seconds = request_timeout_seconds

    def encode(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
    ) -> list[list[float]]:
        """Encode texts through Ollama's local embedding API."""
        vectors = [self._encode_one(text) for text in texts]
        if normalize:
            return [_normalize_vector(vector) for vector in vectors]
        return vectors

    def _encode_one(self, text: str) -> list[float]:
        payload = self._post_json(
            f"{self.base_url}/api/embeddings",
            {
                "model": self.model,
                "prompt": text,
            },
        )
        return _embedding_from_payload(payload)

    def _post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(  # noqa: S310 - expected local Ollama endpoint.
                request,
                timeout=self.request_timeout_seconds,
            ) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            msg = f"Ollama embedding request failed with HTTP {error.code}."
            raise RuntimeError(msg) from error
        except (OSError, json.JSONDecodeError) as error:
            msg = "Ollama embedding request failed."
            raise RuntimeError(msg) from error
        if not isinstance(decoded, dict):
            msg = "Ollama embedding response must be a JSON object."
            raise RuntimeError(msg)
        return decoded


def build_embedding_provider(kind: str = "demo") -> EmbeddingProvider:
    """Build an embedding provider without making network calls up front."""
    normalized_kind = kind.casefold().strip() or "demo"
    if normalized_kind == "demo":
        from hirag_ontology.pipeline.runner import demo_embedding_provider

        return demo_embedding_provider()

    settings = load_embedding_settings()
    provider = settings.provider if normalized_kind == "auto" else normalized_kind
    if provider == "demo":
        from hirag_ontology.pipeline.runner import demo_embedding_provider

        return demo_embedding_provider()
    if provider == "openai":
        return OpenAIEmbeddingProvider(
            model=settings.model,
            base_url=settings.base_url,
            api_key=settings.api_key,
            request_timeout_seconds=settings.request_timeout_seconds,
        )
    if provider == "ollama":
        return OllamaEmbeddingProvider(
            model=settings.model,
            base_url=settings.base_url,
            request_timeout_seconds=settings.request_timeout_seconds,
        )
    msg = "embedding provider must be one of: demo, auto, openai, ollama."
    raise ValueError(msg)


class RetrievalMode(StrEnum):
    """Supported retrieval modes."""

    SEMANTIC_ONLY = "semantic_only"
    LEXICAL_ONLY = "lexical_only"
    LEXICAL_STRUCTURAL = "lexical_structural"
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
        mode: RetrievalMode = RetrievalMode.LEXICAL_STRUCTURAL,
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
        if self.mode == RetrievalMode.LEXICAL_STRUCTURAL:
            ranked, component_scores = self._lexical_structural_scores(query)
            return self._to_results(
                ranked,
                top_k,
                self.mode.value,
                component_scores=component_scores,
            )
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
        query_vector = self.embedding_provider.encode(
            [expand_text_for_retrieval(query)],
            normalize=True,
        )[0]
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
        query_tokens = tokenize(expand_text_for_retrieval(query))
        scores = bm25_scores(query_tokens, tokenized_documents)
        return self._sort_scores(
            [
                (entity_id, score)
                for (entity_id, _), score in zip(entity_items, scores, strict=True)
            ]
        )

    def _lexical_structural_scores(
        self,
        query: str,
    ) -> tuple[list[tuple[str, float]], dict[str, dict[str, float]]]:
        lexical = self._lexical_scores(query)
        structural = self._structural_scores()
        lexical_scores = dict(lexical)
        structural_scores = dict(structural)
        normalized_lexical = _normalize_scores(lexical_scores)
        normalized_structural = _normalize_scores(structural_scores)
        lexical_weight = 0.85
        structural_weight = 0.15
        scores = [
            (
                entity_id,
                (lexical_weight * normalized_lexical.get(entity_id, 0.0))
                + (
                    structural_weight
                    * normalized_structural.get(entity_id, 0.0)
                ),
            )
            for entity_id in self.kg.entities
        ]
        return self._sort_scores(scores), {
            "lexical": lexical_scores,
            "structural": structural_scores,
        }

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
        component_name: str | None = None,
        component_scores: dict[str, dict[str, float]] | None = None,
    ) -> list[RetrievedEntity]:
        return [
            RetrievedEntity(
                entity_id=entity_id,
                entity=self.kg.entities[entity_id],
                score=score,
                rank=rank,
                retrieval_mode=retrieval_mode,
                component_scores=(
                    {
                        name: scores.get(entity_id, 0.0)
                        for name, scores in component_scores.items()
                    }
                    if component_scores is not None
                    else ({component_name: score} if component_name is not None else {})
                ),
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
    raw_document = " ".join(
        part
        for part in [
            entity.label,
            entity.label,
            entity.entity_type,
            entity.description,
            " ".join(entity.aliases),
            " ".join(entity.aliases),
        ]
        if part
    )
    return expand_text_for_retrieval(raw_document)


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Russian/Latin medical text with light normalization."""
    normalized = _normalize_search_text(text)
    return re.findall(r"[a-zа-я0-9]+\+?", normalized)


def expand_text_for_retrieval(text: str) -> str:
    """Expand common oncology synonyms and spelling variants for retrieval."""
    normalized = _normalize_search_text(text)
    expansions: list[str] = []
    for pattern, synonyms in _SYNONYM_RULES:
        if re.search(pattern, normalized):
            expansions.extend(synonyms)
    return " ".join([text, *expansions])


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


def _embedding_from_payload(payload: object) -> list[float]:
    if not isinstance(payload, dict):
        msg = "Embedding payload must be a JSON object."
        raise RuntimeError(msg)
    raw_embedding = payload.get("embedding")
    if not isinstance(raw_embedding, list):
        msg = "Embedding payload must contain an embedding list."
        raise RuntimeError(msg)
    vector: list[float] = []
    for value in raw_embedding:
        if not isinstance(value, int | float):
            msg = "Embedding vector must contain only numbers."
            raise RuntimeError(msg)
        vector.append(float(value))
    return vector


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]


def _normalize_search_text(text: str) -> str:
    normalized = text.casefold().replace("ё", "е")
    normalized = re.sub(
        r"\bbcr\s*(?:::|[-/\\]|\s)\s*abl\s*1?\b",
        " bcr abl bcrabl bcr-abl bcr-abl1 ",
        normalized,
    )
    normalized = re.sub(
        r"\bbcr\s*(?:::|[-/\\])\s*abl",
        " bcr abl bcrabl bcr-abl ",
        normalized,
    )
    normalized = re.sub(r"\bph\s*\+\b", " ph+ ph positive ", normalized)
    normalized = re.sub(r"\bph\s*-\b", " ph- ph negative ", normalized)
    normalized = re.sub(r"[^\wа-яА-ЯёЁ+]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 0:
        return {entity_id: 0.0 for entity_id in scores}
    return {
        entity_id: score / max_score
        for entity_id, score in scores.items()
    }


_SYNONYM_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        r"\bостр\w*\s+лимфобластн\w*\s+лейкоз\w*\b|\bолл\b",
        (
            "олл",
            "острый лимфобластный лейкоз",
            "лимфобластный лейкоз",
            "all",
            "acute lymphoblastic leukemia",
        ),
    ),
    (
        r"\bb\s*[- ]?\s*олл\b|\bв\s*[- ]?\s*олл\b|\bb\s*cell\s+all\b",
        (
            "в олл",
            "в-олл",
            "b-олл",
            "b cell acute lymphoblastic leukemia",
        ),
    ),
    (
        r"\bt\s*[- ]?\s*олл\b|\bt\s*cell\s+all\b",
        (
            "t олл",
            "t-олл",
            "t cell acute lymphoblastic leukemia",
        ),
    ),
    (
        r"\bbcr\s+abl\b|\bbcrabl\b|\bbcr\s+abl1\b",
        (
            "bcr-abl",
            "bcr::abl",
            "bcr::abl1",
            "bcr abl",
            "филадельфийская хромосома",
        ),
    ),
    (
        r"\bph\+\b|ph\s*позитивн\w*|филадельфийск\w*",
        (
            "ph+",
            "ph positive",
            "ph-позитивный",
            "филадельфийская хромосома",
            "bcr-abl",
        ),
    ),
    (
        r"\bингибитор\w*\s+тирозинкиназ\w*\b|\bтки\b|tyrosine\s+kinase",
        (
            "тки",
            "ингибиторы тирозинкиназы",
            "ингибитор тирозинкиназы",
            "tyrosine kinase inhibitor",
            "bcr-abl",
        ),
    ),
    (
        r"\bлеч\w*\b|\bтерап\w*\b|\btreat\w*\b",
        (
            "лечение",
            "терапия",
            "химиотерапия",
            "протокол лечения",
            "treats",
        ),
    ),
    (
        r"\bтгск\b|\bткм\b|трансплантац\w*\s+костн\w*\s+мозг\w*|"
        r"трансплантац\w*\s+гемопоэтическ\w*",
        (
            "тгск",
            "ткм",
            "трансплантация костного мозга",
            "трансплантация гемопоэтических стволовых клеток",
        ),
    ),
)
