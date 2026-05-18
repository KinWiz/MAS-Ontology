import json

from hirag_ontology.evaluation.baseline_eval import run_baseline_eval
from hirag_ontology.evaluation.benchmark import load_benchmark
from hirag_ontology.evaluation.dedup_ablation import LabeledPair, run_dedup_ablation
from hirag_ontology.evaluation.generation_eval import (
    exact_context_recall,
    lexical_relevance,
    run_generation_eval,
)
from hirag_ontology.evaluation.latency_eval import run_latency_eval
from hirag_ontology.evaluation.llm_judge import safe_json_parse
from hirag_ontology.evaluation.retrieval_eval import (
    average_precision_at_k,
    hit_at_k,
    reciprocal_rank,
    run_retrieval_eval,
)
from hirag_ontology.evaluation.run_full_eval import run_full_evaluation
from hirag_ontology.pipeline.knowledge_graph import Entity, KnowledgeGraph
from hirag_ontology.retrieval.retriever import FakeEmbeddingProvider, RetrievalMode


def _evaluation_fixture(tmp_path):
    kg = KnowledgeGraph()
    drug_id = kg.add_entity(
        Entity(
            label="imatinib",
            entity_type="Drug",
            aliases=["Glivec"],
            description="BCR-ABL tyrosine kinase inhibitor",
            source_chunks=["doc1:chunk0"],
        )
    )
    condition_id = kg.add_entity(
        Entity(
            label="Ph+ acute lymphoblastic leukemia",
            entity_type="Condition",
            aliases=["Ph+ ALL"],
            source_chunks=["doc1:chunk1"],
        )
    )
    test_id = kg.add_entity(
        Entity(
            label="RT-PCR",
            entity_type="LabTest",
            aliases=["PCR"],
        )
    )
    kg.add_relation_by_ids(drug_id, "treats", condition_id, confidence=0.95)
    kg.add_relation_by_ids(condition_id, "diagnosed_by", test_id, confidence=0.9)
    kg.compute_pagerank()
    graph_path = tmp_path / "graph.json"
    kg.save(graph_path)

    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "q1",
                        "question": "How is Ph+ ALL treated with Glivec?",
                        "type": "single_entity_lookup",
                        "relevant_entity_labels": [
                            "Ph+ acute lymphoblastic leukemia",
                            "imatinib",
                        ],
                    },
                    {
                        "id": "q2",
                        "question": "How is Ph+ ALL diagnosed?",
                        "type": "relation_inference",
                        "relevant_entity_labels": ["RT-PCR"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return kg, graph_path, gt_path, [drug_id, condition_id, test_id]


def test_retrieval_metric_helpers_match_aliases(tmp_path) -> None:
    kg, _, _, ids = _evaluation_fixture(tmp_path)
    drug_id, condition_id, test_id = ids

    retrieved = [test_id, drug_id, condition_id]

    assert hit_at_k(retrieved, kg, ["Glivec"], k=2) == 1.0
    assert reciprocal_rank(retrieved, kg, ["imatinib"]) == 0.5
    assert average_precision_at_k(
        retrieved,
        kg,
        ["imatinib", "Ph+ ALL"],
        k=3,
    ) == (0.5 + (2 / 3)) / 2


def test_run_retrieval_eval_returns_mode_metrics(tmp_path) -> None:
    _, graph_path, gt_path, _ = _evaluation_fixture(tmp_path)

    results = run_retrieval_eval(
        kg_path=graph_path,
        gt_path=gt_path,
        top_k=3,
        modes=[RetrievalMode.LEXICAL_ONLY],
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert results["per_mode"]["lexical_only"]["n_questions"] == 2
    assert results["per_question"][0]["modes"]["lexical_only"]["top_retrieved"]


def test_run_baseline_eval_compares_expected_systems(tmp_path) -> None:
    _, graph_path, gt_path, _ = _evaluation_fixture(tmp_path)

    results = run_baseline_eval(
        kg_path=graph_path,
        gt_path=gt_path,
        top_k=3,
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert set(results["per_system"]) == {"naive_rag", "hirag", "hirag_ontology"}
    assert results["systems"]["naive_rag"]["retrieval_mode"] == "lexical_only"


def test_generation_eval_uses_deterministic_metrics(tmp_path) -> None:
    kg, graph_path, gt_path, ids = _evaluation_fixture(tmp_path)

    results = run_generation_eval(
        kg_path=graph_path,
        gt_path=gt_path,
        top_k=3,
        n_questions=1,
    )

    assert results["n_questions"] == 1
    assert "faithfulness" in results["summary"]
    assert exact_context_recall(kg, ids[:2], ["Glivec", "RT-PCR"]) == 0.5
    assert lexical_relevance("Ph ALL treatment", "Ph ALL treatment uses imatinib") > 0


def test_latency_eval_returns_stage_breakdown(tmp_path) -> None:
    _, graph_path, gt_path, _ = _evaluation_fixture(tmp_path)

    results = run_latency_eval(
        kg_path=graph_path,
        gt_path=gt_path,
        n_queries=1,
    )

    assert results["n_queries"] == 1
    assert "retrieval_s" in results["aggregated"]
    assert results["per_query"][0]["n_entities_retrieved"] > 0


def test_dedup_ablation_scores_labeled_pairs(tmp_path) -> None:
    _, graph_path, _, _ = _evaluation_fixture(tmp_path)

    results = run_dedup_ablation(
        kg_path=graph_path,
        alphas=[0.6],
        thresholds=[0.8],
        pairs=(
            LabeledPair("alpha beta", "beta alpha", True),
            LabeledPair("alpha beta", "gamma", False),
        ),
    )

    assert results["best"]["precision"] == 1.0
    assert results["best"]["recall"] == 1.0


def test_full_evaluation_writes_artifacts(tmp_path) -> None:
    _, graph_path, gt_path, _ = _evaluation_fixture(tmp_path)
    out_dir = tmp_path / "eval"

    report = run_full_evaluation(
        kg_path=graph_path,
        gt_path=gt_path,
        out_dir=out_dir,
        top_k=3,
        n_latency=1,
        n_generation=1,
        skip_dedup=True,
    )

    assert (out_dir / "full_evaluation_report.json").exists()
    assert (out_dir / "evaluation_report.md").exists()
    assert (out_dir / "retrieval_metrics.json").exists()
    assert (out_dir / "baseline_metrics.json").exists()
    assert report["components"]["retrieval"]
    assert report["components"]["baselines"]


def test_benchmark_loader_and_judge_json_parse(tmp_path) -> None:
    _, _, gt_path, _ = _evaluation_fixture(tmp_path)

    questions = load_benchmark(gt_path)

    assert questions[0].id == "q1"
    assert safe_json_parse("```json\n{\"faithfulness\": 0.75}\n```") == {
        "faithfulness": 0.75
    }


def test_default_benchmark_has_paper_style_size() -> None:
    questions = load_benchmark("evaluation/ground_truth.json")

    assert len(questions) == 50
    assert len({question.id for question in questions}) == 50
