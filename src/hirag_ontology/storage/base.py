"""Shared interfaces and DTOs for graph storage backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from hirag_ontology.pipeline.knowledge_graph import KnowledgeGraph


@dataclass(frozen=True)
class GraphStats:
    """Small storage-neutral graph statistics."""

    entity_count: int
    relation_count: int


@dataclass(frozen=True)
class GraphNode:
    """Storage-neutral node representation for APIs and visualizations."""

    id: str
    label: str
    entity_type: str
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    source_chunks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphRelation:
    """Storage-neutral relation representation for APIs and visualizations."""

    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0
    source_chunk: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphSubgraph:
    """A limited and paginated graph slice."""

    nodes: list[GraphNode]
    relations: list[GraphRelation]
    total_relations: int
    limit: int
    offset: int
    has_more: bool


class GraphStore(Protocol):
    """Minimal persistence interface for graph backends."""

    def read_graph(self) -> KnowledgeGraph:
        """Read a graph from the backend."""

    def write_graph(self, kg: KnowledgeGraph, *, clear: bool = False) -> None:
        """Write a graph to the backend."""

    def stats(self) -> GraphStats:
        """Return graph statistics from the backend."""

    def search_entities(
        self,
        query: str = "",
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[GraphNode]:
        """Search entities by label, aliases, description, or type."""

    def get_subgraph(
        self,
        entity_id: str,
        *,
        depth: int = 1,
        limit: int = 50,
        offset: int = 0,
    ) -> GraphSubgraph:
        """Return a limited subgraph around one entity."""


def normalize_limit(limit: int, *, default: int, maximum: int) -> int:
    """Clamp pagination limits to safe positive bounds."""
    if limit <= 0:
        return default
    return min(limit, maximum)


def normalize_offset(offset: int) -> int:
    """Clamp pagination offsets to zero or greater."""
    return max(offset, 0)


def normalize_depth(depth: int, *, default: int = 1, maximum: int = 2) -> int:
    """Clamp graph traversal depth to visualization-safe bounds."""
    if depth <= 0:
        return default
    return min(depth, maximum)
