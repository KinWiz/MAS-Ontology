"""Optional graph persistence backends."""

from hirag_ontology.storage.base import (
    GraphNode,
    GraphRelation,
    GraphStats,
    GraphStore,
    GraphSubgraph,
)
from hirag_ontology.storage.json_store import JsonGraphStore
from hirag_ontology.storage.neo4j_store import Neo4jGraphStore

__all__ = [
    "GraphNode",
    "GraphRelation",
    "GraphStats",
    "GraphStore",
    "GraphSubgraph",
    "JsonGraphStore",
    "Neo4jGraphStore",
]
