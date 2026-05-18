"""Optional LLM-as-judge helpers for generation evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from hirag_ontology.llm import LLMClient

FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI-generated answer is faithful to the provided
knowledge graph context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Return only JSON:
{{
  "faithfulness": <number from 0.0 to 1.0>,
  "comment": "<short explanation>"
}}
"""


CONTEXT_PRECISION_PROMPT = """\
You are evaluating whether retrieved graph entities are relevant to a question.

Question:
{question}

Retrieved entities:
{entities}

Return only JSON:
{{
  "context_precision": <number from 0.0 to 1.0>,
  "comment": "<short explanation>"
}}
"""


@dataclass(frozen=True)
class JudgeScore:
    """One numeric LLM judge score with a short comment."""

    score: float
    comment: str


def judge_faithfulness(
    llm_client: LLMClient,
    *,
    question: str,
    context: str,
    answer: str,
) -> JudgeScore:
    """Score answer faithfulness using an optional LLM judge."""
    prompt = FAITHFULNESS_PROMPT.format(
        context=context,
        question=question,
        answer=answer,
    )
    payload = _complete_json_object(llm_client, prompt)
    return JudgeScore(
        score=_clamp_float(payload.get("faithfulness", 0.0)),
        comment=str(payload.get("comment", "")),
    )


def judge_context_precision(
    llm_client: LLMClient,
    *,
    question: str,
    retrieved_entities: list[str],
) -> JudgeScore:
    """Score retrieved-entity precision using an optional LLM judge."""
    prompt = CONTEXT_PRECISION_PROMPT.format(
        question=question,
        entities="\n".join(f"- {label}" for label in retrieved_entities),
    )
    payload = _complete_json_object(llm_client, prompt)
    return JudgeScore(
        score=_clamp_float(payload.get("context_precision", 0.0)),
        comment=str(payload.get("comment", "")),
    )


def safe_json_parse(text: str) -> dict[str, Any]:
    """Parse JSON even when an LLM wraps it in markdown code fences."""
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def _complete_json_object(llm_client: LLMClient, prompt: str) -> dict[str, Any]:
    response = llm_client.complete_json(prompt, schema_name="judge")
    if isinstance(response, dict):
        return response
    return safe_json_parse(response)


def _clamp_float(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        return 0.0
    try:
        numeric = float(value)
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, numeric))
