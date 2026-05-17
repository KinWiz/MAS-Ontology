from hirag_ontology.app.web_demo import (
    ask_payload,
    dashboard_payload,
    neo4j_status_payload,
    retrieval_compare_payload,
    search_entities_payload,
    subgraph_payload,
)
from hirag_ontology.pipeline.knowledge_graph import Entity, KnowledgeGraph


def _graph_path(tmp_path):
    kg = KnowledgeGraph()
    drug_id = kg.add_entity(
        Entity(
            label="imatinib",
            entity_type="Drug",
            description="BCR-ABL inhibitor",
            aliases=["Glivec"],
        )
    )
    condition_id = kg.add_entity(
        Entity(
            label="Ph+ acute lymphoblastic leukemia",
            entity_type="Condition",
            aliases=["Ph+ ALL"],
        )
    )
    test_id = kg.add_entity(Entity(label="RT-PCR", entity_type="LabTest"))
    kg.add_relation_by_ids(drug_id, "treats", condition_id, confidence=0.95)
    kg.add_relation_by_ids(condition_id, "diagnosed_by", test_id, confidence=0.9)
    kg.compute_pagerank()
    graph_path = tmp_path / "graph.json"
    kg.save(graph_path)
    return graph_path, condition_id


def test_web_dashboard_payload_contains_counts_and_validation(tmp_path) -> None:
    graph_path, _ = _graph_path(tmp_path)

    payload = dashboard_payload(graph_path)

    assert payload["entity_count"] == 3
    assert payload["relation_count"] == 2
    assert payload["type_distribution"]["Drug"] == 1
    assert payload["validation"]["status"] == "valid"
    assert payload["graph_metrics"]["connected_components"] == 1
    assert payload["graph_metrics"]["isolated_entities"] == 0
    assert payload["top_predicates"][0]["predicate"] in {"diagnosed_by", "treats"}
    assert payload["top_by_degree"]
    assert payload["top_by_pagerank"]


def test_web_entity_search_filters_by_type(tmp_path) -> None:
    graph_path, _ = _graph_path(tmp_path)

    payload = search_entities_payload(
        graph_path=graph_path,
        query="ph+",
        entity_type="Condition",
    )

    assert payload["total"] == 1
    assert payload["items"][0]["label"] == "Ph+ acute lymphoblastic leukemia"


def test_web_subgraph_payload_is_limited_and_clickable(tmp_path) -> None:
    graph_path, condition_id = _graph_path(tmp_path)

    payload = subgraph_payload(
        graph_path=graph_path,
        entity_id=condition_id,
        depth=1,
        limit_nodes=2,
    )

    assert len(payload["nodes"]) == 2
    assert payload["relations"]
    assert any(node["selected"] for node in payload["nodes"])


def test_web_ask_payload_uses_deterministic_mode_without_llm(tmp_path) -> None:
    graph_path, _ = _graph_path(tmp_path)

    payload = ask_payload(
        graph_path=graph_path,
        query="How is Ph+ ALL treated?",
        retrieval_mode="lexical_only",
        top_k=2,
        llm="deterministic",
    )

    assert "answer" in payload
    assert "Ph+ acute lymphoblastic leukemia" in payload["answer"]
    assert payload["retrieved_entities"]
    assert payload["diagnostics"]["retrieval_mode"] == "lexical_only"
    assert payload["context_relations"]
    assert "Graph Context" not in payload["graph_context"]


def test_web_retrieval_compare_payload_returns_modes(tmp_path) -> None:
    graph_path, _ = _graph_path(tmp_path)

    payload = retrieval_compare_payload(
        graph_path=graph_path,
        query="How is Ph+ ALL treated?",
        top_k=2,
    )

    assert "lexical_only" in payload["modes"]
    assert "lexical_structural" in payload["modes"]
    assert payload["modes"]["lexical_only"]["items"]


def test_web_neo4j_status_does_not_require_password(monkeypatch) -> None:
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    payload = neo4j_status_payload()

    assert payload["configured"] is False
    assert payload["connected"] is False
    assert payload["target"]["password_set"] is False
