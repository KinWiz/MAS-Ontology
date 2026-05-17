import pytest

from hirag_ontology.pipeline.chunking import (
    chunk_text,
    document_id_from_path,
    load_markdown_chunks,
)


def _words(count: int) -> str:
    return " ".join(f"w{i}" for i in range(count))


def test_chunk_text_respects_chunk_size() -> None:
    chunks = chunk_text(_words(9), chunk_size=4, overlap=1)

    assert chunks == [
        "w0 w1 w2 w3",
        "w3 w4 w5 w6",
        "w6 w7 w8",
    ]
    assert all(len(chunk.split()) <= 4 for chunk in chunks)


def test_chunk_text_preserves_overlap() -> None:
    chunks = chunk_text(_words(10), chunk_size=5, overlap=2)

    first = chunks[0].split()
    second = chunks[1].split()

    assert first[-2:] == second[:2]


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_text("one two", chunk_size=2, overlap=2)


def test_empty_text_and_empty_markdown_file_emit_no_chunks(tmp_path) -> None:
    assert chunk_text("") == []

    (tmp_path / "empty.md").write_text(" \n\t ", encoding="utf-8")

    assert load_markdown_chunks(tmp_path, chunk_size=5, overlap=1) == []


def test_load_markdown_chunks_preserves_metadata(tmp_path) -> None:
    source = tmp_path / "Guideline A.md"
    source.write_text(_words(7), encoding="utf-8")

    chunks = load_markdown_chunks(tmp_path, chunk_size=4, overlap=1)

    assert [chunk.chunk_id for chunk in chunks] == [
        "guideline_a::chunk-0000",
        "guideline_a::chunk-0001",
    ]
    assert chunks[0].document_id == "guideline_a"
    assert chunks[0].source_path == str(source.resolve())
    assert chunks[0].start_word == 0
    assert chunks[0].end_word == 4
    assert chunks[1].start_word == 3
    assert chunks[1].end_word == 7
    assert chunks[1].text == "w3 w4 w5 w6"


def test_load_markdown_chunks_uses_deterministic_document_order(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (nested / "c.md").write_text("gamma", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("ignored", encoding="utf-8")

    chunks = load_markdown_chunks(tmp_path, chunk_size=10, overlap=0)

    assert [chunk.document_id for chunk in chunks] == ["a", "b", "nested__c"]
    assert [chunk.text for chunk in chunks] == ["alpha", "beta", "gamma"]


def test_document_id_is_based_on_normalized_relative_path(tmp_path) -> None:
    nested = tmp_path / "Clinical Docs"
    nested.mkdir()
    source = nested / "ALL Guideline.md"

    assert document_id_from_path(source, tmp_path) == "clinical_docs__all_guideline"
