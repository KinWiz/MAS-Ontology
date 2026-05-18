import hirag_ontology.cli as cli
from hirag_ontology import __version__
from hirag_ontology.cli import build_parser, main
from hirag_ontology.llm import FakeLLMClient
from hirag_ontology.pipeline.knowledge_graph import Entity, KnowledgeGraph
from hirag_ontology.storage import GraphStats
from test_pipeline_demo import _patch_demo_clients


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"


def test_parser_exposes_run_demo_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args(["run-demo"])

    assert args.command == "run-demo"
    assert args.input == "data/sample_docs"
    assert args.out == "results/demo_graph.json"
    assert args.llm == "gemma"


def test_parser_accepts_gemma_runtime() -> None:
    parser = build_parser()

    run_demo_args = parser.parse_args(["run-demo", "--llm", "gemma"])
    ask_args = parser.parse_args(
        ["ask", "--query", "How is Ph+ ALL treated?", "--llm", "gemma"]
    )

    assert run_demo_args.llm == "gemma"
    assert ask_args.llm == "gemma"


def test_parser_accepts_remote_llm_runtime_choices() -> None:
    parser = build_parser()

    run_demo_args = parser.parse_args(["run-demo", "--llm", "openai"])
    ask_args = parser.parse_args(
        ["ask", "--query", "How is Ph+ ALL treated?", "--llm", "deepseek"]
    )

    assert run_demo_args.llm == "openai"
    assert ask_args.llm == "deepseek"


def test_parser_exposes_ask_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args(["ask", "--query", "How is Ph+ ALL treated?"])

    assert args.command == "ask"
    assert args.graph == "results/demo_graph.json"
    assert args.query == "How is Ph+ ALL treated?"
    assert args.llm == "gemma"
    assert args.top_k == 5
    assert args.retrieval_mode == "lexical_structural"
    assert args.embedding_provider == "demo"
    assert args.show_context is False


def test_parser_exposes_neo4j_export_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args(["export-neo4j"])

    assert args.command == "export-neo4j"
    assert args.graph == "results/knowledge_graph_full_gemma.json"
    assert args.uri is None
    assert args.user is None
    assert args.password is None
    assert args.database is None
    assert args.clear is False


def test_parser_exposes_web_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args(["web"])

    assert args.command == "web"
    assert args.host == "127.0.0.1"
    assert args.port == 8765
    assert args.graph == "results/knowledge_graph_full_gemma.json"


def test_parser_exposes_evaluate_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args(["evaluate"])

    assert args.command == "evaluate"
    assert args.kg == "results/knowledge_graph_full_gemma.json"
    assert args.gt == "evaluation/ground_truth.json"
    assert args.out_dir == "results"
    assert args.top_k == 10
    assert args.n_latency == 20
    assert args.n_generation is None
    assert args.skip_generation is False
    assert args.skip_dedup is False
    assert args.skip_baselines is False
    assert args.apply_dedup_ablation is False
    assert args.embedding_provider == "demo"


def test_graph_stats_command_prints_json_graph_stats(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    kg = KnowledgeGraph()
    kg.add_relation("imatinib", "treats", "Ph-positive ALL")
    kg.save(graph_path)

    exit_code = main(["graph-stats", "--graph", str(graph_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "2 entities, 1 relations" in captured.out


def test_export_neo4j_uses_optional_store_without_logging_password(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    graph_path = tmp_path / "graph.json"
    kg = KnowledgeGraph()
    kg.add_entity(Entity(label="imatinib", entity_type="Drug"))
    kg.save(graph_path)
    instances = []

    class FakeNeo4jStore:
        def __init__(self, **kwargs) -> None:  # noqa: ANN001
            self.kwargs = kwargs
            self.clear = None
            instances.append(self)

        def write_graph(self, graph, *, clear: bool = False) -> None:  # noqa: ANN001
            self.graph = graph
            self.clear = clear

        def stats(self) -> GraphStats:
            return GraphStats(entity_count=1, relation_count=0)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(cli, "Neo4jGraphStore", FakeNeo4jStore)

    exit_code = main(
        [
            "export-neo4j",
            "--graph",
            str(graph_path),
            "--uri",
            "bolt://example:7687",
            "--user",
            "neo4j",
            "--password",
            "secret",
            "--database",
            "neo4j",
            "--clear",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert instances[0].kwargs["uri"] == "bolt://example:7687"
    assert instances[0].clear is True
    assert instances[0].closed is True
    assert "1 entities, 0 relations" in captured.out
    assert "secret" not in captured.out


def test_run_demo_outputs_summary(tmp_path, capsys, monkeypatch) -> None:
    input_dir = tmp_path / "sample_docs"
    input_dir.mkdir()
    (input_dir / "a.md").write_text(
        "imatinib is used in this deterministic Ph+ ALL sample.",
        encoding="utf-8",
    )
    (input_dir / "b.md").write_text(
        "dasatinib is used in this deterministic Ph+ ALL sample.",
        encoding="utf-8",
    )
    output_path = tmp_path / "results" / "demo_graph.json"
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
    assert "Run summary saved:" in captured.out


def test_ask_command_answers_from_saved_demo_graph(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "sample_docs"
    input_dir.mkdir()
    (input_dir / "a.md").write_text(
        "imatinib is used in this deterministic Ph+ ALL sample.",
        encoding="utf-8",
    )
    (input_dir / "b.md").write_text(
        "dasatinib is used in this deterministic Ph+ ALL sample.",
        encoding="utf-8",
    )
    graph_path = tmp_path / "results" / "demo_graph.json"
    _patch_demo_clients(monkeypatch)
    main(
        [
            "run-demo",
            "--input",
            str(input_dir),
            "--out",
            str(graph_path),
            "--llm",
            "gemma",
        ]
    )
    monkeypatch.setattr(
        cli,
        "_build_gemma_answer_client",
        lambda: FakeLLMClient(
            text_responses={
                "You are": "Mock answer about Ph+ acute lymphoblastic leukemia."
            }
        ),
    )

    exit_code = main(
        [
            "ask",
            "--graph",
            str(graph_path),
            "--query",
            "How is Ph+ ALL treated?",
            "--llm",
            "gemma",
            "--show-context",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Answer:" in captured.out
    assert "Mock answer" in captured.out
    assert "Retrieved entities:" in captured.out
    assert "Graph context:" in captured.out
    assert "Ph+ acute lymphoblastic leukemia" in captured.out


def test_ask_missing_graph_returns_error(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "ask",
            "--graph",
            str(tmp_path / "missing.json"),
            "--query",
            "Question",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "error:" in captured.out


def test_run_demo_default_input_resolves_from_project_root(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = tmp_path / "results" / "demo_graph.json"
    _patch_demo_clients(monkeypatch)

    exit_code = main(
        [
            "run-demo",
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
