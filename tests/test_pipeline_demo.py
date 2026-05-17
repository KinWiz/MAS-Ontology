import json

from hirag_ontology.cli import main
from hirag_ontology.llm import FakeLLMClient
from hirag_ontology.pipeline import runner
from hirag_ontology.pipeline.knowledge_graph import KnowledgeGraph


def _write_sample_docs(input_dir) -> None:
    input_dir.mkdir()
    (input_dir / "a.md").write_text(
        "imatinib is used in this deterministic Ph+ ALL sample.",
        encoding="utf-8",
    )
    (input_dir / "b.md").write_text(
        "dasatinib is used in this deterministic Ph+ ALL sample.",
        encoding="utf-8",
    )


def _patch_demo_clients(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "_build_extraction_client",
        lambda llm: FakeLLMClient(json_responses=_extraction_responses()),
    )
    monkeypatch.setattr(
        runner,
        "_build_typing_client",
        lambda llm, extraction_client: FakeLLMClient(
            json_responses=_typing_responses()
        ),
    )


def _extraction_responses() -> dict[str, dict]:
    return {
        "extract": {"entities": [], "relations": []},
        "imatinib": {
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
                    "description": "Philadelphia-positive acute lymphoblastic leukemia",
                    "aliases": ["Ph+ ALL"],
                },
                {
                    "label": "RT-PCR",
                    "type": "LabTest",
                    "description": "Molecular diagnostic test for BCR-ABL transcripts",
                },
                {
                    "label": "FISH",
                    "type": "LabTest",
                    "description": "Cytogenetic test for BCR-ABL rearrangement",
                },
                {
                    "label": "400 mg daily",
                    "type": "DosageRegimen",
                    "description": "Example deterministic dosage regimen",
                },
            ],
            "relations": [
                {
                    "subject": "imatinib",
                    "predicate": "treats",
                    "object": "Ph+ acute lymphoblastic leukemia",
                    "confidence": 0.95,
                },
                {
                    "subject": "Ph+ acute lymphoblastic leukemia",
                    "predicate": "diagnosed_by",
                    "object": "RT-PCR",
                    "confidence": 0.9,
                },
                {
                    "subject": "Ph+ acute lymphoblastic leukemia",
                    "predicate": "diagnosed_by",
                    "object": "FISH",
                    "confidence": 0.88,
                },
                {
                    "subject": "imatinib",
                    "predicate": "dosage_is",
                    "object": "400 mg daily",
                    "confidence": 0.85,
                },
            ],
        },
        "dasatinib": {
            "entities": [
                {
                    "label": "dasatinib",
                    "type": "Drug",
                    "description": (
                        "Second-generation BCR-ABL tyrosine kinase inhibitor"
                    ),
                },
                {
                    "label": "Ph+ acute lymphoblastic leukemia",
                    "type": "Condition",
                    "description": "Philadelphia-positive acute lymphoblastic leukemia",
                    "aliases": ["Ph+ ALL"],
                },
                {
                    "label": "nausea",
                    "type": "Symptom",
                    "description": "Example adverse event",
                },
                {
                    "label": "induction therapy",
                    "type": "Procedure",
                    "description": "Initial treatment phase",
                },
                {
                    "label": "ALL treatment protocol",
                    "type": "Procedure",
                    "description": "Overall acute lymphoblastic leukemia protocol",
                },
            ],
            "relations": [
                {
                    "subject": "dasatinib",
                    "predicate": "treats",
                    "object": "Ph+ acute lymphoblastic leukemia",
                    "confidence": 0.92,
                },
                {
                    "subject": "dasatinib",
                    "predicate": "causes",
                    "object": "nausea",
                    "confidence": 0.7,
                },
                {
                    "subject": "induction therapy",
                    "predicate": "part_of",
                    "object": "ALL treatment protocol",
                    "confidence": 0.8,
                },
            ],
        },
    }


def _typing_responses() -> dict[str, dict]:
    return {
        "type:imatinib": {"class": "Drug", "confidence": 0.98},
        "type:dasatinib": {"class": "Drug", "confidence": 0.98},
        "type:ph+ acute lymphoblastic leukemia": {
            "class": "Condition",
            "confidence": 0.97,
        },
        "type:rt-pcr": {"class": "LabTest", "confidence": 0.95},
        "type:fish": {"class": "LabTest", "confidence": 0.95},
        "type:400 mg daily": {"class": "DosageRegimen", "confidence": 0.92},
        "type:nausea": {"class": "Symptom", "confidence": 0.9},
        "type:induction therapy": {"class": "Procedure", "confidence": 0.9},
        "type:all treatment protocol": {
            "class": "Procedure",
            "confidence": 0.9,
        },
    }


def test_run_demo_pipeline_creates_graph_summary_and_retrieval(
    tmp_path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "sample_docs"
    output_path = tmp_path / "results" / "demo_graph.json"
    _write_sample_docs(input_dir)
    _patch_demo_clients(monkeypatch)

    summary = runner.run_demo_pipeline(input_dir=input_dir, out_path=output_path)

    loaded = KnowledgeGraph.load(output_path)
    summary_payload = json.loads(
        output_path.with_name("run_summary.json").read_text(encoding="utf-8")
    )

    assert output_path.exists()
    assert summary_payload == summary
    assert len(loaded.entities) >= 5
    assert len(loaded.relations) >= 5
    assert loaded.pagerank
    assert summary["documents_processed"] == 2
    assert summary["chunks_processed"] == 2
    assert summary["pagerank_computed"] is True
    assert summary["consistency_final"] == 1.0
    assert summary["reasoning"]["added_relations"] == 1
    assert summary["retrieved_entities"]
    assert any(
        relation.predicate == "related_to"
        and relation.metadata["inferred_by"] == "shared_condition_rule"
        for relation in loaded.relations
    )


def test_cli_run_demo_command_writes_output(tmp_path, capsys, monkeypatch) -> None:
    input_dir = tmp_path / "sample_docs"
    output_path = tmp_path / "results" / "demo_graph.json"
    _write_sample_docs(input_dir)
    _patch_demo_clients(monkeypatch)

    exit_code = main(
        [
            "run-demo",
            "--input",
            str(input_dir),
            "--out",
            str(output_path),
            "--llm",
            "gemma",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    assert "Demo graph saved:" in captured.out
    assert "Top retrieval:" in captured.out
