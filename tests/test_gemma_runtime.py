import json
from typing import Any

import pytest

from hirag_ontology.config import (
    load_deepseek_settings,
    load_gemma_settings,
    load_openai_settings,
)
from hirag_ontology.llm import (
    GemmaOllamaClient,
    LLMResponseError,
    OpenAICompatibleChatClient,
    build_llm_client,
)


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


def test_load_remote_chat_settings_use_safe_defaults(monkeypatch) -> None:
    for name in [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "OPENAI_MAX_RETRIES",
        "OPENAI_MIN_REQUEST_INTERVAL_SECONDS",
        "OPENAI_TEMPERATURE",
        "OPENAI_REQUEST_TIMEOUT_SECONDS",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_MAX_RETRIES",
        "DEEPSEEK_MIN_REQUEST_INTERVAL_SECONDS",
        "DEEPSEEK_TEMPERATURE",
        "DEEPSEEK_REQUEST_TIMEOUT_SECONDS",
    ]:
        monkeypatch.delenv(name, raising=False)

    openai = load_openai_settings(env_path="missing.env")
    deepseek = load_deepseek_settings(env_path="missing.env")

    assert openai.base_url == "https://api.openai.com/v1"
    assert openai.model == "gpt-4o-mini"
    assert openai.api_key == ""
    assert "api_key=<empty>" in repr(openai)
    assert deepseek.base_url == "https://api.deepseek.com"
    assert deepseek.model == "deepseek-chat"
    assert deepseek.temperature is None


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


def test_openai_compatible_client_posts_chat_completion(monkeypatch) -> None:
    payloads: list[dict[str, Any]] = []
    urls: list[str] = []
    auth_headers: list[str | None] = []

    def fake_urlopen(request, timeout: float):  # noqa: ANN001
        assert timeout == 12.0
        urls.append(request.full_url)
        auth_headers.append(request.get_header("Authorization"))
        payloads.append(json.loads(request.data.decode("utf-8")))
        return _FakeHTTPResponse(
            {"choices": [{"message": {"content": "remote answer"}}]}
        )

    monkeypatch.setattr("hirag_ontology.llm.urlopen", fake_urlopen)
    client = OpenAICompatibleChatClient(
        provider_name="openai",
        model="gpt-test",
        api_key="secret-key",
        base_url="https://api.example/v1",
        temperature=0.0,
        min_request_interval_seconds=0,
        request_timeout_seconds=12.0,
    )

    assert client.complete_text("Question") == "remote answer"
    assert urls == ["https://api.example/v1/chat/completions"]
    assert auth_headers == ["Bearer secret-key"]
    assert payloads[0]["model"] == "gpt-test"
    assert payloads[0]["messages"][0]["content"] == "Question"
    assert payloads[0]["temperature"] == 0.0
    assert "response_format" not in payloads[0]


def test_openai_compatible_client_complete_json_uses_json_object_mode(
    monkeypatch,
) -> None:
    payloads: list[dict[str, Any]] = []

    def fake_urlopen(request, timeout: float):  # noqa: ANN001
        del timeout
        payloads.append(json.loads(request.data.decode("utf-8")))
        return _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "{\"class\": \"Drug\", \"confidence\": 0.8}"
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("hirag_ontology.llm.urlopen", fake_urlopen)
    client = OpenAICompatibleChatClient(
        provider_name="deepseek",
        model="deepseek-test",
        api_key="secret-key",
        base_url="https://api.deepseek.test",
        min_request_interval_seconds=0,
    )

    result = client.complete_json("Return JSON")

    assert result == {"class": "Drug", "confidence": 0.8}
    assert payloads[0]["response_format"] == {"type": "json_object"}


def test_build_llm_client_uses_remote_env_without_network(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")

    client = build_llm_client("openai")

    assert isinstance(client, OpenAICompatibleChatClient)
    assert client.provider_name == "openai"
    assert client.model == "gpt-test"
    assert client.base_url == "https://api.example/v1"
