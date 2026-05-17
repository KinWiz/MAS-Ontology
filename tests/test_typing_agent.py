import hashlib
import json

from hirag_ontology.llm import FakeLLMClient
from hirag_ontology.pipeline.knowledge_graph import (
    Entity,
    KnowledgeGraph,
    normalize_label,
)
from hirag_ontology.pipeline.typing_agent import TypingAgent


def test_typing_agent_assigns_valid_ontology_class(tmp_path) -> None:
    client = FakeLLMClient(
        json_responses={
            "type:imatinib": {
                "class": "Drug",
                "confidence": 0.95,
                "rationale": "The label refers to a medication.",
            }
        }
    )
    agent = TypingAgent(client, cache_dir=tmp_path)
    entity = Entity(
        label="imatinib",
        description="BCR-ABL tyrosine kinase inhibitor",
    )

    result = agent.type_entity(entity)

    assert result.assigned_class == "Drug"
    assert result.confidence == 0.95
    assert result.rationale == "The label refers to a medication."
    assert result.cache_hit is False
    assert entity.entity_type == "Drug"
    assert client.json_calls[0]["schema_name"] == "type:imatinib"
    assert "Drug" in client.json_calls[0]["prompt"]
    assert "BCR-ABL tyrosine kinase inhibitor" in client.json_calls[0]["prompt"]


def test_typing_agent_falls_back_to_other_for_invalid_class(tmp_path) -> None:
    client = FakeLLMClient(
        json_responses={
            "type:imatinib": {
                "class": "Medication",
                "confidence": 0.8,
                "rationale": "Close but not an ontology class.",
            }
        }
    )
    agent = TypingAgent(client, cache_dir=tmp_path)
    entity = Entity(label="imatinib")

    result = agent.type_entity(entity)

    assert result.assigned_class == "Other"
    assert result.confidence == 0.8
    assert result.raw["class"] == "Medication"
    assert entity.entity_type == "Other"


def test_typing_agent_uses_cache_by_normalized_entity_label(tmp_path) -> None:
    client = FakeLLMClient(
        json_responses={
            "type:imatinib": {
                "class": "Drug",
                "confidence": 0.95,
                "rationale": "The label refers to a medication.",
            }
        }
    )
    agent = TypingAgent(client, cache_dir=tmp_path)

    first = agent.type_entity(Entity(label="  Imatinib "))
    second = agent.type_entity(Entity(label="imatinib"))

    normalized = normalize_label("Imatinib")
    cache_name = hashlib.md5(normalized.encode("utf-8")).hexdigest() + ".json"
    cached_payload = json.loads((tmp_path / cache_name).read_text(encoding="utf-8"))

    assert cached_payload["class"] == "Drug"
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.assigned_class == "Drug"
    assert len(client.json_calls) == 1


def test_type_graph_updates_entities_and_node_metadata(tmp_path) -> None:
    client = FakeLLMClient(
        json_responses={
            "type:imatinib": {"class": "Drug", "confidence": 0.95},
            "type:ph+ acute lymphoblastic leukemia": {
                "class": "Condition",
                "confidence": 0.9,
            },
        }
    )
    agent = TypingAgent(client, cache_dir=tmp_path)
    kg = KnowledgeGraph()
    drug_id = kg.add_entity(Entity(label="imatinib"))
    condition_id = kg.add_entity(Entity(label="Ph+ acute lymphoblastic leukemia"))

    stats = agent.type_graph(kg)

    assert stats == {
        "typed_count": 2,
        "cache_hits": 0,
        "cache_misses": 2,
        "assigned_counts": {"Drug": 1, "Condition": 1},
    }
    assert kg.entities[drug_id].entity_type == "Drug"
    assert kg.entities[condition_id].entity_type == "Condition"
    assert kg.graph.nodes[drug_id]["entity_type"] == "Drug"
    assert kg.graph.nodes[condition_id]["entity_type"] == "Condition"
