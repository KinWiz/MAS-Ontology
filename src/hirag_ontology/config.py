"""Runtime configuration loaded from environment variables and .env files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GemmaSettings:
    """Local Gemma 4 runtime settings for Ollama."""

    model: str
    base_url: str
    max_retries: int
    min_request_interval_seconds: float
    temperature: float
    request_timeout_seconds: float

    def __repr__(self) -> str:
        return (
            "GemmaSettings("
            f"model={self.model!r}, "
            f"base_url={self.base_url!r}, "
            f"max_retries={self.max_retries!r}, "
            f"min_request_interval_seconds={self.min_request_interval_seconds!r}, "
            f"temperature={self.temperature!r}, "
            f"request_timeout_seconds={self.request_timeout_seconds!r})"
        )


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if key in os.environ and not override:
            continue
        os.environ[key] = _clean_env_value(value)


def load_gemma_settings(env_path: str | Path = ".env") -> GemmaSettings:
    """Load local Gemma 4 settings from .env and the current environment."""
    load_dotenv(env_path)
    base_url = os.getenv("GEMMA_BASE_URL", "http://localhost:11434").strip()
    model = os.getenv("GEMMA_MODEL", "gemma4:latest").strip() or "gemma4:latest"
    return GemmaSettings(
        model=model,
        base_url=base_url.rstrip("/"),
        max_retries=_get_int_env("GEMMA_MAX_RETRIES", default=2),
        min_request_interval_seconds=_get_float_env(
            "GEMMA_MIN_REQUEST_INTERVAL_SECONDS",
            default=0.5,
        ),
        temperature=_get_float_env("GEMMA_TEMPERATURE", default=0.0),
        request_timeout_seconds=_get_float_env(
            "GEMMA_REQUEST_TIMEOUT_SECONDS",
            default=600.0,
        ),
    )


def _clean_env_value(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        return cleaned[1:-1]
    return cleaned


def _get_int_env(name: str, *, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        msg = f"{name} must be an integer."
        raise ValueError(msg) from error


def _get_float_env(name: str, *, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError as error:
        msg = f"{name} must be a number."
        raise ValueError(msg) from error
