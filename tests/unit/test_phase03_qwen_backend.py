"""Unit tests for the Qwen API backend (env, config, response, factory).

Real API network calls are NOT tested here — they run via
scripts/data/smoke_qwen_surface_realizer.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from homechef_booking.inference.factory import load_qwen_surface_backend
from homechef_booking.inference.qwen_api_backend import (
    QwenApiBackend,
    QwenApiConfig,
    check_qwen_env,
)


class TestEnvFailFast:
    """DASHSCOPE env var handling (no fallback)."""

    def test_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.com/v1")
        cfg = QwenApiConfig(model="qwen-plus")
        with pytest.raises(OSError) as exc:
            QwenApiBackend(cfg)
        assert "DASHSCOPE_API_KEY_MISSING" in str(exc.value)

    def test_base_url_missing(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key-not-real")
        cfg = QwenApiConfig(model="qwen-plus")
        with pytest.raises(OSError) as exc:
            QwenApiBackend(cfg)
        assert "DASHSCOPE_BASE_URL_MISSING" in str(exc.value)

    def test_model_not_configured(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key-not-real")
        monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.com/v1")
        cfg = QwenApiConfig(model="")  # empty model
        with pytest.raises(ValueError) as exc:
            QwenApiBackend(cfg)
        assert "QWEN_MODEL_NOT_CONFIGURED" in str(exc.value)

    def test_env_status_reports_set_missing(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
        monkeypatch.setenv("DASHSCOPE_BASE_URL", "u")
        assert check_qwen_env() == ("SET", "SET")

        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        assert check_qwen_env()[0] == "MISSING"
        assert check_qwen_env()[1] == "SET"


class TestConfigParsing:
    """QwenApiConfig field validation."""

    def test_defaults(self):
        cfg = QwenApiConfig(model="qwen-plus")
        assert cfg.temperature == 0.8
        assert cfg.max_tokens == 512
        assert cfg.timeout_s == 180.0
        assert cfg.api_key_env == "DASHSCOPE_API_KEY"
        assert cfg.base_url_env == "DASHSCOPE_BASE_URL"

    def test_extra_forbidden(self):
        with pytest.raises(ValueError):
            QwenApiConfig(model="m", extra_field="x")


class TestChatUrl:
    """URL construction from base_url."""

    def test_chat_url_v1(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
        monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        backend = QwenApiBackend(QwenApiConfig(model="qwen-plus"))
        assert backend._chat_url() == (
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        )

    def test_chat_url_no_v1(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
        monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.com/api")
        backend = QwenApiBackend(QwenApiConfig(model="qwen-plus"))
        assert backend._chat_url() == "https://example.com/api/chat/completions"


class TestResponseExtraction:
    """Parsing of OpenAI-compatible chat response (no network)."""

    def test_extract_content(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
        monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.com/v1")
        backend = QwenApiBackend(QwenApiConfig(model="qwen-plus"))

        data = {"choices": [{"message": {"content": "  改写结果。  "}}]}
        content = backend._parse_chat_response(data)
        assert content == "改写结果。"

    def test_malformed_response_raises(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
        monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.com/v1")
        backend = QwenApiBackend(QwenApiConfig(model="qwen-plus"))
        with pytest.raises(RuntimeError):
            backend._parse_chat_response({"choices": []})


@pytest.fixture
def workspace_tmp(tmp_path_factory):
    """Workspace-local temp dir (avoids OneDrive/Temp permission issues)."""
    from pathlib import Path as _Path
    base = _Path(__file__).resolve().parent.parent.parent / ".pytest_tmp_qwen"
    base.mkdir(parents=True, exist_ok=True)
    return base


class TestFactoryRegistration:
    """load_qwen_surface_backend reads qwen_surface.yaml via the factory."""

    def _write_config(self, tmp_path: Path, model: str) -> Path:
        p = tmp_path / "qwen_surface.yaml"
        p.write_text(
            f"backend: qwen_api\n"
            f"model: {model or '""'}\n"
            f"temperature: 0.8\n"
            f"max_tokens: 512\n"
            f"timeout_s: 180\n"
            f"api_key_env: DASHSCOPE_API_KEY\n"
            f"base_url_env: DASHSCOPE_BASE_URL\n",
            encoding="utf-8",
        )
        return p

    def test_factory_rejects_wrong_backend(self, workspace_tmp):
        p = workspace_tmp / "x.yaml"
        p.write_text("backend: mock\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_qwen_surface_backend(p)

    def test_factory_loads_qwen_backend(self, workspace_tmp, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
        monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.com/v1")
        p = self._write_config(workspace_tmp, model="qwen-plus")
        backend = load_qwen_surface_backend(p)
        assert isinstance(backend, QwenApiBackend)
        assert backend.config.model == "qwen-plus"
