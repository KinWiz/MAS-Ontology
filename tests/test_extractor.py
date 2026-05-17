import hashlib
import json

import pytest

from hirag_ontology.llm import FakeLLMClient, LLMResponseError
from hirag_ontology.pipeline.chunking import TextChunk
from hirag_ontology.pipeline.extractor import ExtractionAgent
from hirag_ontology.pipeline.knowledge_graph import KnowledgeGraph, entity_id_from_label


def _extract_response() -> dict[str, object]:
    return {
        "entities": [
            {
                "label": "imatinib",
                "type": "Drug",
                "description": "BCR-ABL tyrosine kinase inhibitor",
                "aliases": ["Glivec"],
            },
            {
                "label": "Ph+ acute lymphoblastic leukemia",
                "type": "Condition",
                "description": "Philadelphia-positive ALL",
            },
        ],
        "relations": [
            {
                "subject": "imatinib",
                "predicate": "treats",
                "object": "Ph+ acute lymphoblastic leukemia",
                "confidence": 0.9,
            }
        ],
    }


def _chunk() -> TextChunk:
    return TextChunk(
        document_id="doc",
        chunk_id="doc::chunk-0000",
        text="Imatinib treats Ph+ acute lymphoblastic leukemia.",
        source_path="doc.md",
        start_word=0,
        end_word=6,
    )


def test_fake_llm_client_returns_schema_response() -> None:
    client = FakeLLMClient(
        json_responses={"extract": {"entities": [], "relations": []}}
    )

    response = client.complete_json("prompt", schema_name="extract")

    assert response == {"entities": [], "relations": []}
    assert client.json_calls == [{"prompt": "prompt", "schema_name": "extract"}]


def test_extraction_agent_parses_entities_and_relations(tmp_path) -> None:
    client = FakeLLMClient(json_responses={"extract": _extract_response()})
    agent = ExtractionAgent(client, cache_dir=tmp_path)

    result = agent.extract(_chunk())

    assert [entity.label for entity in result.entities] == [
        "imatinib",
        "Ph+ acute lymphoblastic leukemia",
    ]
    assert result.entities[0].entity_type == "Drug"
    assert result.entities[0].aliases == ["Glivec"]
    assert result.entities[0].source_chunks == ["doc::chunk-0000"]
    assert result.relations[0].subject == "imatinib"
    assert result.relations[0].predicate == "treats"
    assert result.relations[0].object == "Ph+ acute lymphoblastic leukemia"
    assert result.relations[0].confidence == 0.9
    assert result.relations[0].source_chunk == "doc::chunk-0000"


def test_extraction_agent_populates_knowledge_graph(tmp_path) -> None:
    client = FakeLLMClient(json_responses={"extract": _extract_response()})
    agent = ExtractionAgent(client, cache_dir=tmp_path)
    kg = KnowledgeGraph()

    agent.extract_to_graph(kg, _chunk())

    imatinib_id = entity_id_from_label("imatinib")
    condition_id = entity_id_from_label("Ph+ acute lymphoblastic leukemia")
    assert kg.entities[imatinib_id].entity_type == "Drug"
    assert kg.entities[condition_id].entity_type == "Condition"
    assert kg.graph.has_edge(imatinib_id, condition_id)
    assert kg.graph.edges[imatinib_id, condition_id]["predicate"] == "treats"


def test_extraction_agent_caches_by_md5_chunk_text(tmp_path) -> None:
    client = FakeLLMClient(json_responses={"extract": _extract_response()})
    chunk = _chunk()
    agent = ExtractionAgent(client, cache_dir=tmp_path)

    first = agent.extract(chunk)
    second = agent.extract(chunk)

    cache_name = hashlib.md5(chunk.text.encode("utf-8")).hexdigest() + ".json"
    cached_payload = json.loads((tmp_path / cache_name).read_text(encoding="utf-8"))
    assert cached_payload == _extract_response()
    assert first.entities == second.entities
    assert len(client.json_calls) == 1


def test_extraction_agent_accepts_raw_json_string(tmp_path) -> None:
    client = FakeLLMClient(json_responses={"extract": json.dumps(_extract_response())})
    agent = ExtractionAgent(client, cache_dir=tmp_path)

    result = agent.extract(_chunk())

    assert result.entities[0].label == "imatinib"


def test_extraction_agent_rejects_malformed_json(tmp_path) -> None:
    client = FakeLLMClient(json_responses={"extract": "{not-valid-json"})
    agent = ExtractionAgent(client, cache_dir=tmp_path)

    with pytest.raises(LLMResponseError, match="Malformed JSON"):
        agent.extract(_chunk())


def test_extraction_agent_rejects_invalid_response_shape(tmp_path) -> None:
    client = FakeLLMClient(
        json_responses={"extract": {"entities": {}, "relations": []}}
    )
    agent = ExtractionAgent(client, cache_dir=tmp_path)

    with pytest.raises(LLMResponseError, match="entities"):
        agent.extract(_chunk())
