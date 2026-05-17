import pytest

from hirag_ontology.pipeline.knowledge_graph import (
    Entity,
    KnowledgeGraph,
    entity_id_from_label,
)
from hirag_ontology.retrieval.retriever import (
    FakeEmbeddingProvider,
    HybridRetriever,
    RetrievalMode,
    bm25_scores,
    expand_text_for_retrieval,
    tokenize,
)
from hirag_ontology.retrieval.rrf import rrf_fusion


def _kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    imatinib_id = kg.add_entity(
        Entity(
            label="imatinib",
            entity_type="Drug",
            description="BCR-ABL tyrosine kinase inhibitor",
            aliases=["Glivec"],
        )
    )
    dasatinib_id = kg.add_entity(
        Entity(
            label="dasatinib",
            entity_type="Drug",
            description="second generation tyrosine kinase inhibitor",
        )
    )
    condition_id = kg.add_entity(
        Entity(
            label="Ph+ acute lymphoblastic leukemia",
            entity_type="Condition",
            description="Philadelphia-positive acute leukemia",
            aliases=["Ph+ ALL"],
        )
    )
    lab_id = kg.add_entity(
        Entity(
            label="RT-PCR",
            entity_type="LabTest",
            description="BCR-ABL molecular diagnostic test",
        )
    )
    kg.add_relation_by_ids(imatinib_id, "treats", condition_id)
    kg.add_relation_by_ids(dasatinib_id, "treats", condition_id)
    kg.add_relation_by_ids(condition_id, "diagnosed_by", lab_id)
    return kg


def _embedding_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(
        {
            "leukemia treatment": [1.0, 0.0, 0.0],
            "imatinib": [1.0, 0.0, 0.0],
            "dasatinib": [0.9, 0.1, 0.0],
            "ph+ acute lymphoblastic leukemia": [0.8, 0.2, 0.0],
            "rt-pcr": [0.0, 1.0, 0.0],
        }
    )


def test_rrf_fusion_uses_one_based_ranks_and_k() -> None:
    fused = rrf_fusion([["a", "b"], ["b", "c"]], k=60)

    assert fused[0][0] == "b"
    assert fused[0][1] == pytest.approx((1 / 62) + (1 / 61))
    assert fused[1][0] == "a"
    assert fused[1][1] == pytest.approx(1 / 61)
    assert fused[2][0] == "c"
    assert fused[2][1] == pytest.approx(1 / 62)


def test_bm25_scores_are_deterministic() -> None:
    scores = bm25_scores(
        ["glivec", "inhibitor"],
        [
            ["imatinib", "glivec", "inhibitor"],
            ["rt-pcr", "diagnostic", "test"],
        ],
    )

    assert scores[0] > scores[1]
    assert scores == bm25_scores(
        ["glivec", "inhibitor"],
        [
            ["imatinib", "glivec", "inhibitor"],
            ["rt-pcr", "diagnostic", "test"],
        ],
    )


def test_semantic_only_ranking_is_deterministic() -> None:
    retriever = HybridRetriever(
        _kg(),
        _embedding_provider(),
        mode=RetrievalMode.SEMANTIC_ONLY,
    )

    results = retriever.retrieve("leukemia treatment", top_k=3)

    assert [result.entity.label for result in results] == [
        "imatinib",
        "dasatinib",
        "Ph+ acute lymphoblastic leukemia",
    ]
    assert results[0].component_scores["semantic"] == pytest.approx(1.0)


def test_lexical_only_ranking_uses_label_alias_description_and_type() -> None:
    retriever = HybridRetriever(
        _kg(),
        _embedding_provider(),
        mode=RetrievalMode.LEXICAL_ONLY,
    )

    results = retriever.retrieve("Glivec inhibitor", top_k=2)

    assert [result.entity.label for result in results] == ["imatinib", "dasatinib"]
    assert results[0].score > results[1].score


def test_tokenize_expands_russian_oncology_synonyms() -> None:
    tokens = tokenize(
        expand_text_for_retrieval(
            "лечение острого лимфобластного лейкоза BCR ABL "
            "ингибиторы тирозинкиназы"
        )
    )

    assert "олл" in tokens
    assert "bcr" in tokens
    assert "abl" in tokens
    assert "тки" in tokens
    assert "химиотерапия" in tokens


def test_lexical_structural_prioritizes_russian_medical_matches() -> None:
    kg = KnowledgeGraph()
    oll_id = kg.add_entity(
        Entity(
            label="Острый лимфобластный лейкоз (ОЛЛ)",
            entity_type="Condition",
            aliases=["ОЛЛ"],
        )
    )
    tki_id = kg.add_entity(
        Entity(
            label="ингибиторы тирозинкиназы BCR-ABL",
            entity_type="Drug",
            aliases=["ТКИ"],
        )
    )
    unrelated_id = kg.add_entity(
        Entity(label="Рак билиарного тракта", entity_type="Condition")
    )
    kg.add_relation_by_ids(tki_id, "treats", oll_id)
    kg.add_relation_by_ids(unrelated_id, "related_to", oll_id)

    retriever = HybridRetriever(
        kg,
        _embedding_provider(),
        mode=RetrievalMode.LEXICAL_STRUCTURAL,
    )

    results = retriever.retrieve(
        "лечение острого лимфобластного лейкоза BCR ABL "
        "ингибиторы тирозинкиназы",
        top_k=2,
    )

    assert [result.entity.label for result in results] == [
        "ингибиторы тирозинкиназы BCR-ABL",
        "Острый лимфобластный лейкоз (ОЛЛ)",
    ]
    assert results[0].retrieval_mode == "lexical_structural"
    assert set(results[0].component_scores) == {"lexical", "structural"}


def test_structural_only_ranking_uses_pagerank() -> None:
    kg = _kg()
    retriever = HybridRetriever(
        kg,
        _embedding_provider(),
        mode=RetrievalMode.STRUCTURAL_ONLY,
    )

    results = retriever.retrieve("anything", top_k=2)

    assert results[0].entity.label == "RT-PCR"
    assert set(kg.pagerank) == set(kg.entities)


def test_hybrid_rrf_combines_component_rankings() -> None:
    retriever = HybridRetriever(
        _kg(),
        _embedding_provider(),
        mode=RetrievalMode.HYBRID_RRF,
        rrf_k=60,
    )

    results = retriever.retrieve("leukemia treatment", top_k=3)

    assert [result.entity.label for result in results] == [
        "Ph+ acute lymphoblastic leukemia",
        "dasatinib",
        "imatinib",
    ]
    assert results[0].retrieval_mode == "hybrid_rrf"
    assert set(results[0].component_scores) == {"semantic", "lexical", "structural"}


def test_retrieve_respects_top_k_and_empty_graph() -> None:
    retriever = HybridRetriever(
        _kg(),
        _embedding_provider(),
        mode=RetrievalMode.SEMANTIC_ONLY,
    )

    assert len(retriever.retrieve("leukemia treatment", top_k=2)) == 2
    assert retriever.retrieve("leukemia treatment", top_k=0) == []

    empty_retriever = HybridRetriever(KnowledgeGraph(), _embedding_provider())
    assert empty_retriever.retrieve("anything") == []


def test_fake_embedding_provider_uses_configured_vectors() -> None:
    provider = FakeEmbeddingProvider({"imatinib": [3.0, 4.0]})

    vector = provider.encode(["imatinib"], normalize=True)[0]

    assert vector == pytest.approx([0.6, 0.8])
    assert provider.calls == [["imatinib"]]


def test_retrieved_entity_ids_are_expected() -> None:
    retriever = HybridRetriever(
        _kg(),
        _embedding_provider(),
        mode=RetrievalMode.LEXICAL_ONLY,
    )

    results = retriever.retrieve("Ph+ ALL", top_k=1)

    assert results[0].entity_id == entity_id_from_label(
        "Ph+ acute lymphoblastic leukemia"
    )
