"""LLM client abstractions used by pipeline agents."""

from __future__ import annotations

import copy
import json
import logging
import time
from collections.abc import Mapping
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hirag_ontology.config import (
    ChatAPISettings,
    load_deepseek_settings,
    load_gemma_settings,
    load_openai_settings,
)

JsonResponse = dict[str, Any] | str
logger = logging.getLogger(__name__)
SUPPORTED_LLM_BACKENDS = ("gemma", "openai", "deepseek")
SUPPORTED_ANSWER_BACKENDS = (*SUPPORTED_LLM_BACKENDS, "deterministic")


class LLMError(RuntimeError):
    """Base error for LLM client failures."""


class LLMResponseError(LLMError):
    """Raised when an LLM response cannot be parsed as expected."""


class LLMClient(Protocol):
    """Minimal interface required by pipeline agents."""

    def complete_json(
        self,
        prompt: str,
        *,
        schema_name: str | None = None,
    ) -> JsonResponse:
        """Return a JSON-compatible response for a prompt."""

    def complete_text(self, prompt: str) -> str:
        """Return a text completion for a prompt."""


class FakeLLMClient:
    """Deterministic in-memory LLM client for tests and demos."""

    def __init__(
        self,
        *,
        json_responses: Mapping[str, JsonResponse] | None = None,
        text_responses: Mapping[str, str] | None = None,
    ) -> None:
        self.json_responses = dict(json_responses or {})
        self.text_responses = dict(text_responses or {})
        self.json_calls: list[dict[str, str | None]] = []
        self.text_calls: list[str] = []

    def complete_json(
        self,
        prompt: str,
        *,
        schema_name: str | None = None,
    ) -> JsonResponse:
        """Return a configured JSON fixture by schema name or prompt content."""
        self.json_calls.append({"prompt": prompt, "schema_name": schema_name})
        response = self._lookup(self.json_responses, prompt, schema_name)
        return copy.deepcopy(response)

    def complete_text(self, prompt: str) -> str:
        """Return a configured text fixture by prompt content."""
        self.text_calls.append(prompt)
        return str(self._lookup(self.text_responses, prompt, None))

    @staticmethod
    def _lookup(
        responses: Mapping[str, JsonResponse] | Mapping[str, str],
        prompt: str,
        schema_name: str | None,
    ) -> JsonResponse | str:
        if prompt in responses:
            return responses[prompt]

        for key, response in responses.items():
            if schema_name is not None and key == schema_name:
                continue
            if key in prompt:
                return response

        if schema_name is not None and schema_name in responses:
            return responses[schema_name]

        known = ", ".join(sorted(responses)) or "<none>"
        msg = (
            f"No fake LLM response configured for schema={schema_name!r}; "
            f"known={known}"
        )
        raise KeyError(msg)


class OpenAICompatibleChatClient:
    """Chat completions client for OpenAI-compatible remote APIs."""

    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float | None = None,
        max_retries: int = 2,
        min_request_interval_seconds: float = 0.5,
        request_timeout_seconds: float = 120.0,
    ) -> None:
        if not api_key:
            msg = f"{provider_name} API key is required."
            raise ValueError(msg)
        self.provider_name = provider_name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_request_interval_seconds = min_request_interval_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self._last_request_time = 0.0

    def complete_json(
        self,
        prompt: str,
        *,
        schema_name: str | None = None,
    ) -> dict[str, Any]:
        """Return a parsed JSON object from a remote chat completion."""
        del schema_name
        text = self._complete(prompt, json_mode=True)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            msg = f"{self.provider_name} response was not valid JSON."
            raise LLMResponseError(msg) from error
        if not isinstance(payload, dict):
            msg = f"{self.provider_name} JSON response must be an object."
            raise LLMResponseError(msg)
        return payload

    def complete_text(self, prompt: str) -> str:
        """Return plain text from a remote chat completion."""
        return self._complete(prompt, json_mode=False)

    def _complete(
        self,
        prompt: str,
        *,
        json_mode: bool,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self._wait_for_rate_limit()
                payload: dict[str, Any] = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
                if self.temperature is not None:
                    payload["temperature"] = self.temperature
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}

                response = self._post_chat(payload)
                content = _chat_completion_content(response)
                if content is None:
                    msg = (
                        f"{self.provider_name} response did not include "
                        "message content."
                    )
                    raise LLMResponseError(msg)
                return content
            except Exception as error:  # pragma: no cover - optional runtime path
                last_error = error
                if attempt >= self.max_retries:
                    break
                logger.warning(
                    "%s request attempt %s/%s failed with %s; retrying.",
                    self.provider_name,
                    attempt + 1,
                    self.max_retries + 1,
                    type(error).__name__,
                )
                time.sleep(0.5 * (2**attempt))

        msg = f"{self.provider_name} request failed after retries."
        raise LLMError(msg) from last_error

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            detail = self._redact(error.read().decode("utf-8", errors="replace"))
            msg = f"{self.provider_name} returned HTTP {error.code}: {detail}"
            raise LLMError(msg) from error
        except URLError as error:
            msg = f"Could not connect to {self.provider_name} at {self.base_url}."
            raise LLMError(msg) from error

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as error:
            msg = f"{self.provider_name} response was not valid JSON."
            raise LLMResponseError(msg) from error
        if not isinstance(parsed, dict):
            msg = f"{self.provider_name} response must be a JSON object."
            raise LLMResponseError(msg)
        return parsed

    def _wait_for_rate_limit(self) -> None:
        if self.min_request_interval_seconds <= 0:
            self._last_request_time = time.monotonic()
            return
        now = time.monotonic()
        elapsed = now - self._last_request_time
        wait_seconds = self.min_request_interval_seconds - elapsed
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        self._last_request_time = time.monotonic()

    def _redact(self, text: str) -> str:
        return text.replace(self.api_key, "<redacted>")


class GemmaOllamaClient:
    """Local Gemma 4 chat client using Ollama."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_retries: int = 2,
        min_request_interval_seconds: float = 0.5,
        request_timeout_seconds: float = 600.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_request_interval_seconds = min_request_interval_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self._last_request_time = 0.0

    def complete_json(
        self,
        prompt: str,
        *,
        schema_name: str | None = None,
    ) -> dict[str, Any]:
        """Return a parsed JSON object from local Gemma 4."""
        del schema_name
        text = self._complete(prompt, json_mode=True)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            msg = "Gemma 4 response was not valid JSON."
            raise LLMResponseError(msg) from error
        if not isinstance(payload, dict):
            msg = "Gemma 4 JSON response must be an object."
            raise LLMResponseError(msg)
        return payload

    def complete_text(self, prompt: str) -> str:
        """Return plain text from local Gemma 4."""
        return self._complete(prompt, json_mode=False)

    def _complete(
        self,
        prompt: str,
        *,
        json_mode: bool,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self._wait_for_rate_limit()
                payload: dict[str, Any] = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": self.temperature},
                }
                if json_mode:
                    payload["format"] = "json"

                response = self._post_chat(payload)
                message = response.get("message", {})
                content = message.get("content") if isinstance(message, dict) else None
                if content is None:
                    msg = "Gemma 4 response did not include message content."
                    raise LLMResponseError(msg)
                return str(content)
            except Exception as error:  # pragma: no cover - optional runtime path
                last_error = error
                if attempt >= self.max_retries:
                    break
                logger.warning(
                    "Gemma 4 request attempt %s/%s failed with %s; retrying.",
                    attempt + 1,
                    self.max_retries + 1,
                    type(error).__name__,
                )
                time.sleep(0.5 * (2**attempt))

        msg = (
            "Gemma 4 request failed after retries. "
            "Make sure Ollama is running and the configured Gemma 4 model is pulled."
        )
        raise LLMError(msg) from last_error

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            msg = f"Ollama returned HTTP {error.code}: {detail}"
            raise LLMError(msg) from error
        except URLError as error:
            msg = f"Could not connect to Ollama at {self.base_url}."
            raise LLMError(msg) from error

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as error:
            msg = "Ollama response was not valid JSON."
            raise LLMResponseError(msg) from error
        if not isinstance(parsed, dict):
            msg = "Ollama response must be a JSON object."
            raise LLMResponseError(msg)
        return parsed

    def _wait_for_rate_limit(self) -> None:
        if self.min_request_interval_seconds <= 0:
            self._last_request_time = time.monotonic()
            return
        now = time.monotonic()
        elapsed = now - self._last_request_time
        wait_seconds = self.min_request_interval_seconds - elapsed
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        self._last_request_time = time.monotonic()


def build_llm_client(llm: str) -> LLMClient:
    """Build an LLM client for the requested runtime backend."""
    backend = normalize_llm_backend(llm)
    if backend == "gemma":
        settings = load_gemma_settings()
        logger.info(
            "Using local Gemma 4 runtime: model=%s base_url=%s",
            settings.model,
            settings.base_url,
        )
        return GemmaOllamaClient(
            model=settings.model,
            base_url=settings.base_url,
            temperature=settings.temperature,
            max_retries=settings.max_retries,
            min_request_interval_seconds=settings.min_request_interval_seconds,
            request_timeout_seconds=settings.request_timeout_seconds,
        )
    if backend == "openai":
        return _build_openai_compatible_client(
            load_openai_settings(),
            api_key_env="OPENAI_API_KEY",
        )
    if backend == "deepseek":
        return _build_openai_compatible_client(
            load_deepseek_settings(),
            api_key_env="DEEPSEEK_API_KEY",
        )

    msg = f"llm must be one of: {', '.join(SUPPORTED_LLM_BACKENDS)}."
    raise ValueError(msg)


def normalize_llm_backend(llm: str) -> str:
    """Normalize user-provided LLM backend identifiers."""
    return llm.strip().casefold()


def _build_openai_compatible_client(
    settings: ChatAPISettings,
    *,
    api_key_env: str,
) -> OpenAICompatibleChatClient:
    if not settings.api_key:
        msg = f"{api_key_env} is required to use --llm {settings.provider}."
        raise ValueError(msg)
    logger.info(
        "Using %s chat API runtime: model=%s base_url=%s api_key=%s",
        settings.provider,
        settings.model,
        settings.base_url,
        "<set>",
    )
    return OpenAICompatibleChatClient(
        provider_name=settings.provider,
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=settings.temperature,
        max_retries=settings.max_retries,
        min_request_interval_seconds=settings.min_request_interval_seconds,
        request_timeout_seconds=settings.request_timeout_seconds,
    )


def _chat_completion_content(response: dict[str, Any]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if content is None:
        return None
    return str(content)
