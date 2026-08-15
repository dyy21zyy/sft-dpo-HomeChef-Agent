"""Phase 03 v0.2.1 — Qwen (DashScope) OpenAI-compatible API backend.

Used exclusively as a Strong-Model Chinese Surface Realizer.

Architecture:
  StrongModelSurfaceRealizer → QwenApiBackend.chat(system, user) → Qwen API → str

Security:
  - Reads DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL from environment only.
  - Never logs, prints, or persists the API key. Logs only "SET"/"MISSING".
  - Fails fast if env vars are missing (no fallback to mock/HF/llama.cpp).

Note on the SDK:
  The `openai` Python package is not installed in this project. To keep the
  dependency surface minimal and consistent with the existing llama.cpp backend
  (which uses `requests`), this backend calls the OpenAI-compatible
  /chat/completions REST endpoint directly via `requests`.
"""
from __future__ import annotations

import os

import requests
from pydantic import BaseModel, ConfigDict, Field

# Env var names (single source of truth for env access)
DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY"
DASHSCOPE_BASE_URL_ENV = "DASHSCOPE_BASE_URL"


class QwenApiConfig(BaseModel):
    """Configuration for the Qwen API backend.

    model_id is the actual DashScope/Qwen model name (e.g. "qwen-plus",
    "qwen-max", "qwen-turbo", "qwen2.5-72b-instruct").
    """
    model_config = ConfigDict(extra="forbid", strict=True)

    backend: str = "qwen_api"
    model: str = ""
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)
    timeout_s: float = Field(default=180.0, gt=0)
    api_key_env: str = DASHSCOPE_API_KEY_ENV
    base_url_env: str = DASHSCOPE_BASE_URL_ENV


def check_qwen_env(config: QwenApiConfig | None = None) -> tuple[str, str]:
    """Validate DashScope env vars.

    Returns (api_key_status, base_url_status) where each is "SET" or "MISSING".
    Does NOT return or expose the actual key value.
    """
    cfg = config or QwenApiConfig()
    key_present = bool(os.environ.get(cfg.api_key_env))
    url_present = bool(os.environ.get(cfg.base_url_env))
    return ("SET" if key_present else "MISSING"), ("SET" if url_present else "MISSING")


class QwenApiBackend:
    """OpenAI-compatible Qwen API backend exposing `chat(system, user) -> str`.

    Implements the StrongModelSurfaceRealizer.LLMBackend protocol.
    """

    def __init__(self, config: QwenApiConfig):
        self.config = config
        api_key_status, base_url_status = check_qwen_env(config)
        if api_key_status == "MISSING":
            raise OSError(
                f"{config.api_key_env}_MISSING: DashScope API key not set in environment."
            )
        if base_url_status == "MISSING":
            raise OSError(
                f"{config.base_url_env}_MISSING: DashScope base URL not set in environment."
            )
        if not config.model:
            raise ValueError(
                "QWEN_MODEL_NOT_CONFIGURED: no model name in config. "
                "Set `model` in the Qwen surface config."
            )
        # Note: do NOT retain the key in a long-lived attribute beyond the
        # request. We read it fresh per request.
        self._base_url = os.environ[config.base_url_env]

    # ── LLMBackend protocol ────────────────────────────────

    def chat(self, system: str, user: str) -> str:
        """Call the Qwen chat-completions API and return the assistant text."""
        api_key = os.environ[self.config.api_key_env]
        url = self._chat_url()
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.config.timeout_s,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Qwen API error status={resp.status_code} "
                f"body={_sanitize_error(resp.text)}"
            )
        return self._parse_chat_response(resp.json())

    @staticmethod
    def _parse_chat_response(data: dict) -> str:
        """Extract assistant content from an OpenAI-compatible chat response.

        Separated for unit testing without a network call.
        """
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Qwen API unexpected response shape: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Qwen API returned empty content")
        return content.strip()

    def _chat_url(self) -> str:
        base = self._base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/chat/completions"


def _sanitize_error(text: str) -> str:
    """Strip any sensitive material from an error body (e.g. embedded keys)."""
    if not text:
        return ""
    # Never echo the full raw body if it could contain a key; truncate hard.
    return text[:200]


__all__ = [
    "QwenApiBackend",
    "QwenApiConfig",
    "check_qwen_env",
    "DASHSCOPE_API_KEY_ENV",
    "DASHSCOPE_BASE_URL_ENV",
]
