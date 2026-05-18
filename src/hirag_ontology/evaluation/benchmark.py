"""Benchmark dataset loading helpers for evaluation scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_GROUND_TRUTH_PATH = Path("evaluation") / "ground_truth.json"


@dataclass(frozen=True)
class BenchmarkQuestion:
    """One manually annotated benchmark question."""

    id: str
    question: str
    question_type: str
    relevant_entity_labels: list[str]


def load_benchmark(
    path: str | Path = DEFAULT_GROUND_TRUTH_PATH,
) -> list[BenchmarkQuestion]:
    """Load a ground-truth question set from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_questions = payload.get("questions", [])
    if not isinstance(raw_questions, list):
        msg = "ground_truth.json must contain a questions list."
        raise ValueError(msg)
    return [
        _question_from_payload(index, item)
        for index, item in enumerate(raw_questions)
    ]


def _question_from_payload(index: int, payload: Any) -> BenchmarkQuestion:
    if not isinstance(payload, dict):
        msg = f"question #{index + 1} must be an object."
        raise ValueError(msg)
    labels = payload.get("relevant_entity_labels", [])
    if not isinstance(labels, list) or not all(
        isinstance(item, str) for item in labels
    ):
        msg = f"question #{index + 1} relevant_entity_labels must be a string list."
        raise ValueError(msg)
    return BenchmarkQuestion(
        id=str(payload.get("id", f"q{index + 1:03d}")),
        question=str(payload.get("question", "")).strip(),
        question_type=str(payload.get("type", "unknown")).strip() or "unknown",
        relevant_entity_labels=list(labels),
    )
