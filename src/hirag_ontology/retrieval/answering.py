"""Question answering utilities over retrieved graph context."""

from __future__ import annotations

from hirag_ontology.llm import LLMClient
from hirag_ontology.pipeline.knowledge_graph import KnowledgeGraph, Relation
from hirag_ontology.retrieval.retriever import RetrievedEntity


def build_graph_context(
    kg: KnowledgeGraph,
    retrieved: list[RetrievedEntity],
    *,
    include_neighbors: bool = True,
    max_relations: int = 50,
) -> str:
    """Convert retrieved entities and local relations into LLM context."""
    ordered_entity_ids = _context_entity_ids(
        kg,
        retrieved,
        include_neighbors=include_neighbors,
    )
    context_entity_ids = set(ordered_entity_ids)
    relations = _context_relations(
        kg,
        context_entity_ids,
        max_relations=max_relations,
    )

    entity_lines = [
        _format_entity_line(kg, entity_id)
        for entity_id in ordered_entity_ids
    ]
    relation_lines = [
        _format_relation_line(kg, relation)
        for relation in relations
    ]

    return "\n".join(
        [
            "Entities:",
            *(entity_lines or ["- <none>"]),
            "",
            "Relations:",
            *(relation_lines or ["- <none>"]),
        ]
    )


def build_answer_prompt(query: str, graph_context: str) -> str:
    """Build a safe answer-generation prompt."""
    return "\n".join(
        [
            "You are a medical knowledge graph assistant.",
            "Use only the provided graph context.",
            "If the context is insufficient, say that the answer is not "
            "supported by the graph context.",
            "Do not invent unsupported medical claims.",
            "Answer in the same language as the user's question.",
            "",
            f"Question: {query}",
            "",
            graph_context,
        ]
    )


def answer_from_graph_context(
    llm_client: LLMClient,
    *,
    query: str,
    graph_context: str,
) -> str:
    """Generate an answer from graph context using an LLM client."""
    return llm_client.complete_text(build_answer_prompt(query, graph_context)).strip()


def deterministic_answer_from_graph_context(
    *,
    query: str,
    graph_context: str,
    retrieved: list[RetrievedEntity],
) -> str:
    """Return a deterministic answer for tests."""
    use_russian = _contains_cyrillic(query)
    if not retrieved:
        if use_russian:
            return (
                "Графовый контекст не содержит релевантных сущностей. "
                "Ответ не поддержан графовым контекстом."
            )
        return (
            "The graph context did not retrieve relevant entities. "
            "The answer is not supported by the graph context."
        )

    labels = ", ".join(result.entity.label for result in retrieved)
    relation_lines = [
        line[2:]
        for line in graph_context.splitlines()
        if line.startswith("- ") and " --" in line
    ][:5]
    facts = "; ".join(relation_lines)
    if not facts:
        facts = (
            "локальные связи графа не найдены"
            if use_russian
            else "no local graph relations were found"
        )

    if use_russian:
        return (
            "Только по графовому контексту релевантные сущности: "
            f"{labels}. Поддержанные факты графа: {facts}. "
            "Контекста может быть недостаточно для полного клинического "
            "протокола; детали за пределами этих фактов не поддержаны."
        )

    return (
        "Based only on the graph context, the relevant entities are: "
        f"{labels}. Supported graph facts: {facts}. "
        "The context may be insufficient for a complete clinical protocol; "
        "details beyond these graph facts are not supported."
    )


def _contains_cyrillic(text: str) -> bool:
    return any(
        "а" <= character.lower() <= "я" or character.lower() == "ё"
        for character in text
    )


def _context_entity_ids(
    kg: KnowledgeGraph,
    retrieved: list[RetrievedEntity],
    *,
    include_neighbors: bool,
) -> list[str]:
    ordered_entity_ids: list[str] = []
    seen: set[str] = set()

    for result in retrieved:
        if result.entity_id not in seen:
            ordered_entity_ids.append(result.entity_id)
            seen.add(result.entity_id)

    if include_neighbors:
        for result in retrieved:
            for neighbor_id in kg.neighbors(result.entity_id):
                if neighbor_id not in seen:
                    ordered_entity_ids.append(neighbor_id)
                    seen.add(neighbor_id)

    return ordered_entity_ids


def _context_relations(
    kg: KnowledgeGraph,
    context_entity_ids: set[str],
    *,
    max_relations: int,
) -> list[Relation]:
    if max_relations <= 0:
        return []

    relations: list[Relation] = []
    for relation in kg.relations:
        if (
            relation.subject_id in context_entity_ids
            and relation.object_id in context_entity_ids
        ):
            relations.append(relation)
            if len(relations) >= max_relations:
                break
    return relations


def _format_entity_line(kg: KnowledgeGraph, entity_id: str) -> str:
    entity = kg.entities[entity_id]
    description = f": {entity.description}" if entity.description else ""
    aliases = f" Aliases: {', '.join(entity.aliases)}" if entity.aliases else ""
    return f"- {entity.label} [{entity.entity_type}]{description}.{aliases}".rstrip()


def _format_relation_line(kg: KnowledgeGraph, relation: Relation) -> str:
    subject = kg.entities[relation.subject_id].label
    obj = kg.entities[relation.object_id].label
    return (
        f"- {subject} --{relation.predicate}--> {obj} "
        f"(confidence={relation.confidence:.2f})"
    )
