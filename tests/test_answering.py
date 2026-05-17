from hirag_ontology.llm import FakeLLMClient
from hirag_ontology.pipeline.knowledge_graph import Entity, KnowledgeGraph
from hirag_ontology.retrieval.answering import (
    answer_from_graph_context,
    build_answer_prompt,
    build_graph_context,
    deterministic_answer_from_graph_context,
)
from hirag_ontology.retrieval.retriever import RetrievedEntity


def _graph() -> tuple[KnowledgeGraph, list[RetrievedEntity]]:
    kg = KnowledgeGraph()
    drug_id = kg.add_entity(
        Entity(
            label="imatinib",
            entity_type="Drug",
            description="BCR-ABL tyrosine kinase inhibitor",
            aliases=["Glivec"],
        )
    )
    condition_id = kg.add_entity(
        Entity(
            label="Ph+ acute lymphoblastic leukemia",
            entity_type="Condition",
            description="Philadelphia-positive ALL",
        )
    )
    test_id = kg.add_entity(Entity(label="RT-PCR", entity_type="LabTest"))
    kg.add_relation_by_ids(drug_id, "treats", condition_id, confidence=0.95)
    kg.add_relation_by_ids(condition_id, "diagnosed_by", test_id, confidence=0.9)
    retrieved = [
        RetrievedEntity(
            entity_id=condition_id,
            entity=kg.entities[condition_id],
            score=1.0,
            rank=1,
            retrieval_mode="hybrid_rrf",
        )
    ]
    return kg, retrieved


def test_build_graph_context_includes_entities_neighbors_and_relations() -> None:
    kg, retrieved = _graph()

    context = build_graph_context(kg, retrieved)

    assert "Ph+ acute lymphoblastic leukemia [Condition]" in context
    assert "imatinib [Drug]" in context
    assert "RT-PCR [LabTest]" in context
    assert "imatinib --treats--> Ph+ acute lymphoblastic leukemia" in context
    assert "Ph+ acute lymphoblastic leukemia --diagnosed_by--> RT-PCR" in context


def test_build_graph_context_respects_max_relations() -> None:
    kg, retrieved = _graph()

    context = build_graph_context(kg, retrieved, max_relations=1)

    relation_lines = [
        line
        for line in context.splitlines()
        if line.startswith("- ") and " --" in line
    ]
    assert len(relation_lines) == 1


def test_answer_prompt_contains_safety_instructions() -> None:
    prompt = build_answer_prompt("What is supported?", "Entities:\n- x")

    assert "Use only the provided graph context." in prompt
    assert "Do not invent unsupported medical claims." in prompt
    assert "Question: What is supported?" in prompt


def test_answer_from_graph_context_uses_llm_client() -> None:
    client = FakeLLMClient(
        text_responses={
            "Use only the provided graph context.": "Supported answer.",
        }
    )

    answer = answer_from_graph_context(
        client,
        query="What is supported?",
        graph_context="Entities:\n- imatinib [Drug].",
    )

    assert answer == "Supported answer."
    assert client.text_calls


def test_deterministic_answer_reports_insufficient_context_without_results() -> None:
    answer = deterministic_answer_from_graph_context(
        query="Question",
        graph_context="Entities:\n- <none>",
        retrieved=[],
    )

    assert "not supported by the graph context" in answer


def test_deterministic_answer_lists_entities_and_supported_facts() -> None:
    kg, retrieved = _graph()
    context = build_graph_context(kg, retrieved)

    answer = deterministic_answer_from_graph_context(
        query="Question",
        graph_context=context,
        retrieved=retrieved,
    )

    assert "Ph+ acute lymphoblastic leukemia" in answer
    assert "imatinib --treats--> Ph+ acute lymphoblastic leukemia" in answer


def test_deterministic_answer_uses_russian_for_cyrillic_query() -> None:
    kg, retrieved = _graph()
    context = build_graph_context(kg, retrieved)

    answer = deterministic_answer_from_graph_context(
        query="Как лечат Ph+ ALL?",
        graph_context=context,
        retrieved=retrieved,
    )

    assert "Только по графовому контексту" in answer
    assert "детали за пределами этих фактов не поддержаны" in answer
