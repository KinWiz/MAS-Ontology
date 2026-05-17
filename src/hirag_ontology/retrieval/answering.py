"""Question answering utilities over retrieved graph context."""

from __future__ import annotations

import json
import re
from typing import Any

from hirag_ontology.llm import LLMClient
from hirag_ontology.pipeline.knowledge_graph import KnowledgeGraph, Relation
from hirag_ontology.retrieval.retriever import RetrievedEntity


def build_graph_context(
    kg: KnowledgeGraph,
    retrieved: list[RetrievedEntity],
    *,
    include_neighbors: bool = True,
    max_relations: int = 50,
    max_entities: int = 40,
    query: str | None = None,
) -> str:
    """Convert retrieved entities and local relations into LLM context."""
    ordered_entity_ids = _context_entity_ids(
        kg,
        retrieved,
        include_neighbors=include_neighbors,
        max_entities=max_entities,
        query=query,
    )
    context_entity_ids = set(ordered_entity_ids)
    seed_entity_ids = {result.entity_id for result in retrieved}
    relations = _context_relations(
        kg,
        context_entity_ids,
        seed_entity_ids=seed_entity_ids,
        max_relations=max_relations,
        query=query,
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
            "Return a concise natural-language answer, not JSON.",
            "Do not extract entities or return NER labels.",
            "Do not output markdown tables or raw arrays.",
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
    raw_answer = llm_client.complete_text(
        build_answer_prompt(query, graph_context)
    ).strip()
    return sanitize_answer_text(raw_answer, query=query)


def sanitize_answer_text(answer: str, *, query: str) -> str:
    """Convert accidental JSON extraction output into readable text."""
    stripped = _remove_clinician_irrelevant_disclaimers(answer).strip()
    if not stripped:
        return stripped
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped

    extracted = _extract_json_answer_items(payload)
    if not extracted:
        return stripped

    items = ", ".join(extracted)
    if _contains_cyrillic(query):
        return (
            "Модель вернула список сущностей вместо текстового ответа. "
            f"По графовому контексту релевантны: {items}. "
            "Для полного ответа нужно расширить или уточнить графовый контекст."
        )
    return (
        "The model returned an entity list instead of a textual answer. "
        f"Relevant graph-context items: {items}. "
        "A full answer requires more specific graph context."
    )


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
                "Ответ не поддерживается графовым контекстом."
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
    max_entities: int,
    query: str | None,
) -> list[str]:
    ordered_entity_ids: list[str] = []
    seen: set[str] = set()

    for result in retrieved:
        if result.entity_id not in seen:
            ordered_entity_ids.append(result.entity_id)
            seen.add(result.entity_id)

    if include_neighbors:
        preferred_predicates = _preferred_predicates(query)
        seed_entity_ids = set(ordered_entity_ids)
        neighbor_candidates: list[tuple[int, str, str]] = []
        for relation in kg.relations:
            if (
                relation.subject_id in seed_entity_ids
                and relation.object_id not in seen
            ):
                neighbor_candidates.append(
                    (
                        _predicate_rank(relation.predicate, preferred_predicates),
                        kg.entities[relation.object_id].label.casefold(),
                        relation.object_id,
                    )
                )
            if (
                relation.object_id in seed_entity_ids
                and relation.subject_id not in seen
            ):
                neighbor_candidates.append(
                    (
                        _predicate_rank(relation.predicate, preferred_predicates),
                        kg.entities[relation.subject_id].label.casefold(),
                        relation.subject_id,
                    )
                )

        for _, _, entity_id in sorted(neighbor_candidates):
            if entity_id in seen:
                continue
            ordered_entity_ids.append(entity_id)
            seen.add(entity_id)
            if len(ordered_entity_ids) >= max_entities:
                break

    return ordered_entity_ids


def _context_relations(
    kg: KnowledgeGraph,
    context_entity_ids: set[str],
    *,
    seed_entity_ids: set[str],
    max_relations: int,
    query: str | None,
) -> list[Relation]:
    if max_relations <= 0:
        return []

    relations = [
        relation
        for relation in kg.relations
        if (
            relation.subject_id in context_entity_ids
            and relation.object_id in context_entity_ids
        )
    ]
    preferred_predicates = _preferred_predicates(query)
    relations.sort(
        key=lambda relation: (
            _predicate_rank(relation.predicate, preferred_predicates),
            not (
                relation.subject_id in seed_entity_ids
                or relation.object_id in seed_entity_ids
            ),
            kg.entities[relation.subject_id].label.casefold(),
            relation.predicate.casefold(),
            kg.entities[relation.object_id].label.casefold(),
        )
    )
    return relations[:max_relations]


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


def _preferred_predicates(query: str | None) -> list[str]:
    normalized = (query or "").casefold().replace("ё", "е")
    if any(term in normalized for term in ["леч", "терап", "протокол", "treat"]):
        return [
            "treats",
            "dosage_is",
            "part_of",
            "related_to",
            "contraindicated_for",
            "causes",
            "diagnosed_by",
        ]
    if any(term in normalized for term in ["диагност", "обслед", "анализ", "test"]):
        return [
            "diagnosed_by",
            "related_to",
            "part_of",
            "treats",
            "dosage_is",
            "causes",
            "contraindicated_for",
        ]
    return [
        "treats",
        "diagnosed_by",
        "dosage_is",
        "part_of",
        "related_to",
        "causes",
        "contraindicated_for",
    ]


def _predicate_rank(predicate: str, preferred_predicates: list[str]) -> int:
    try:
        return preferred_predicates.index(predicate)
    except ValueError:
        return len(preferred_predicates)


def _extract_json_answer_items(payload: Any) -> list[str]:
    values: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in ["answer", "text", "name"]:
                item = value.get(key)
                if isinstance(item, str) and item.strip():
                    values.append(item.strip())
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    unique_values: list[str] = []
    seen = set()
    for value in values:
        normalized = value.casefold()
        if normalized not in seen:
            unique_values.append(value)
            seen.add(normalized)
    return unique_values[:8]


def _remove_clinician_irrelevant_disclaimers(answer: str) -> str:
    """Remove patient-facing disclaimers from clinician-oriented answers."""
    cleaned = answer
    disclaimer_patterns = [
        (
            r"\s*Обратите внимание,\s*что\s+это\s+не\s+является\s+"
            r"медицинской\s+консультацией\.?"
        ),
        r"\s*Это\s+не\s+медицинская\s+рекомендация\.?",
        r"\s*Это\s+не\s+является\s+медицинской\s+консультацией\.?",
        r"\s*This\s+is\s+not\s+medical\s+advice\.?",
    ]
    for pattern in disclaimer_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()
