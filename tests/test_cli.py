import hirag_ontology.cli as cli
from hirag_ontology import __version__
from hirag_ontology.cli import build_parser, main
from hirag_ontology.llm import FakeLLMClient
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


def test_parser_exposes_ask_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args(["ask", "--query", "How is Ph+ ALL treated?"])

    assert args.command == "ask"
    assert args.graph == "results/demo_graph.json"
    assert args.query == "How is Ph+ ALL treated?"
    assert args.llm == "gemma"
    assert args.top_k == 5
    assert args.retrieval_mode == "hybrid_rrf"
    assert args.show_context is False


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
