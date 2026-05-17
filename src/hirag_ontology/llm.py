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

JsonResponse = dict[str, Any] | str
logger = logging.getLogger(__name__)


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
