"""Markdown ingestion and deterministic word-level chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextChunk:
    """A text chunk with stable document and source metadata."""

    document_id: str
    chunk_id: str
    text: str
    source_path: str
    start_word: int
    end_word: int


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into deterministic overlapping word chunks."""
    _validate_chunk_settings(chunk_size, overlap)

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def load_markdown_chunks(
    directory: str | Path,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[TextChunk]:
    """Load all Markdown files from a directory into deterministic chunks."""
    _validate_chunk_settings(chunk_size, overlap)

    root = Path(directory)
    if not root.exists():
        msg = f"Markdown directory does not exist: {root}"
        raise FileNotFoundError(msg)
    if not root.is_dir():
        msg = f"Markdown input path is not a directory: {root}"
        raise NotADirectoryError(msg)

    chunks: list[TextChunk] = []
    for path in _iter_markdown_files(root):
        document_id = document_id_from_path(path, root)
        text = path.read_text(encoding="utf-8")
        words = text.split()
        step = chunk_size - overlap

        for index, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start : start + chunk_size]
            if not chunk_words:
                continue

            end = start + len(chunk_words)
            chunks.append(
                TextChunk(
                    document_id=document_id,
                    chunk_id=f"{document_id}::chunk-{index:04d}",
                    text=" ".join(chunk_words),
                    source_path=str(path.resolve()),
                    start_word=start,
                    end_word=end,
                )
            )
            if end >= len(words):
                break

    return chunks


def document_id_from_path(path: str | Path, root: str | Path) -> str:
    """Create a stable document ID from a Markdown path relative to its root."""
    source_path = Path(path)
    root_path = Path(root)
    relative = source_path.relative_to(root_path)
    without_suffix = relative.with_suffix("")
    raw_id = "__".join(without_suffix.parts).casefold()
    normalized = re.sub(r"\W+", "_", raw_id, flags=re.UNICODE).strip("_")
    return normalized or "document"


def _iter_markdown_files(root: Path) -> list[Path]:
    files = [path for path in root.rglob("*") if path.is_file()]
    markdown_files = [path for path in files if path.suffix.casefold() == ".md"]
    return sorted(
        markdown_files,
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def _validate_chunk_settings(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        msg = "chunk_size must be positive"
        raise ValueError(msg)
    if overlap < 0:
        msg = "overlap must be non-negative"
        raise ValueError(msg)
    if overlap >= chunk_size:
        msg = "overlap must be smaller than chunk_size"
        raise ValueError(msg)
