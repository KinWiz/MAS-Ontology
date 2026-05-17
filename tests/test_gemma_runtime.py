import json
from typing import Any

import pytest

from hirag_ontology.config import load_gemma_settings
from hirag_ontology.llm import GemmaOllamaClient, LLMResponseError


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_load_gemma_settings_uses_local_defaults(monkeypatch) -> None:
    monkeypatch.delenv("GEMMA_BASE_URL", raising=False)
    monkeypatch.delenv("GEMMA_MODEL", raising=False)
    monkeypatch.delenv("GEMMA_MAX_RETRIES", raising=False)
    monkeypatch.delenv("GEMMA_MIN_REQUEST_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("GEMMA_TEMPERATURE", raising=False)
    monkeypatch.delenv("GEMMA_REQUEST_TIMEOUT_SECONDS", raising=False)

    settings = load_gemma_settings(env_path="missing.env")

    assert settings.base_url == "http://localhost:11434"
    assert settings.model == "gemma4:latest"
    assert settings.max_retries == 2
    assert settings.min_request_interval_seconds == 0.5
    assert settings.temperature == 0.0
    assert settings.request_timeout_seconds == 600.0


def test_gemma_client_complete_text_posts_to_local_ollama(monkeypatch) -> None:
    payloads: list[dict[str, Any]] = []

    def fake_urlopen(request, timeout: float):  # noqa: ANN001
        assert timeout == 42.0
        payloads.append(json.loads(request.data.decode("utf-8")))
        return _FakeHTTPResponse({"message": {"content": "answer"}})

    monkeypatch.setattr("hirag_ontology.llm.urlopen", fake_urlopen)
    client = GemmaOllamaClient(
        model="gemma4:e4b",
        min_request_interval_seconds=0,
        request_timeout_seconds=42.0,
    )

    assert client.complete_text("Question") == "answer"
    assert payloads[0]["model"] == "gemma4:e4b"
    assert payloads[0]["messages"][0]["content"] == "Question"
    assert "format" not in payloads[0]


def test_gemma_client_complete_json_uses_json_mode(monkeypatch) -> None:
    payloads: list[dict[str, Any]] = []

    def fake_urlopen(request, timeout: int):  # noqa: ANN001
        del timeout
        payloads.append(json.loads(request.data.decode("utf-8")))
        return _FakeHTTPResponse(
            {"message": {"content": '{"class": "Drug", "confidence": 0.9}'}}
        )

    monkeypatch.setattr("hirag_ontology.llm.urlopen", fake_urlopen)
    client = GemmaOllamaClient(
        model="gemma4:e4b",
        min_request_interval_seconds=0,
    )

    result = client.complete_json("Return JSON")

    assert result == {"class": "Drug", "confidence": 0.9}
    assert payloads[0]["format"] == "json"


def test_gemma_client_rejects_malformed_json(monkeypatch) -> None:
    def fake_urlopen(request, timeout: int):  # noqa: ANN001
        del request, timeout
        return _FakeHTTPResponse({"message": {"content": "not-json"}})

    monkeypatch.setattr("hirag_ontology.llm.urlopen", fake_urlopen)
    client = GemmaOllamaClient(
        model="gemma4:e4b",
        min_request_interval_seconds=0,
    )

    with pytest.raises(LLMResponseError, match="Gemma 4 response"):
        client.complete_json("Return JSON")
