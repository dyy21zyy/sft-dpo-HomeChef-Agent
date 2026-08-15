from __future__ import annotations

import json
from pathlib import Path

import yaml

from homechef_booking.inference.backend import Backend
from homechef_booking.inference.hf_backend import HFBackendConfig, HFTransformersBackend
from homechef_booking.inference.llama_cpp_backend import LlamaCppServerBackend, LlamaCppServerConfig
from homechef_booking.inference.mock_backend import MockBackend
from homechef_booking.inference.phase04_hf_backend import (
    Phase04HFConfig,
    Phase04HFTransformersBackend,
)
from homechef_booking.inference.qwen_api_backend import QwenApiBackend, QwenApiConfig


def load_backend(config_path: Path, predictions_path: Path | None = None) -> Backend:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    backend_type = config.get("backend", "mock")
    if backend_type == "mock":
        if predictions_path is not None:
            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        elif "predictions_path" in config:
            predictions_path = Path(config["predictions_path"])
            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        else:
            predictions = config.get("predictions", {})
        return MockBackend(predictions=predictions)
    if backend_type == "hf_transformers":
        hf_config = HFBackendConfig(
            model_id=config.get("model_id", "Qwen/Qwen3-0.6B-Base"),
            device=config.get("device", "cpu"),
            torch_dtype=config.get("torch_dtype", "float32"),
            max_model_length=config.get("max_model_length", 2048),
            max_new_tokens=config.get("max_new_tokens", 512),
            timeout_seconds=config.get("timeout_seconds"),
        )
        backend = HFTransformersBackend()
        backend.load(hf_config)
        return backend
    if backend_type == "phase04_hf":
        p4_config = Phase04HFConfig(
            model_id=config.get("model_id", ""),
            adapter_name_or_path=config.get("adapter_name_or_path"),
            device=config.get("device", "auto"),
            torch_dtype=config.get("torch_dtype", "bfloat16"),
            max_model_length=config.get("max_model_length", 2048),
            max_new_tokens=config.get("max_new_tokens", 512),
            timeout_seconds=config.get("timeout_seconds"),
            use_structured_output=config.get("use_structured_output", False),
            model_key=config.get("model_key"),
            training_stage=config.get("training_stage"),
            model_size=config.get("model_size"),
            pref_beta=config.get("pref_beta"),
        )
        backend = Phase04HFTransformersBackend()
        backend.load(p4_config)
        return backend
    if backend_type == "llama_cpp_server":
        llama_config = LlamaCppServerConfig(
            base_url=config.get("base_url", "http://127.0.0.1:8080/v1"),
            model_id=config.get("model_id", "Qwen/Qwen3-0.6B-Base"),
            max_new_tokens=config.get("max_new_tokens", 512),
            temperature=config.get("temperature", 0.0),
            timeout_seconds=config.get("timeout_seconds", 300.0),
            use_structured_output=config.get("use_structured_output", False),
            device=config.get("device", "cpu"),
            runtime=config.get("runtime", "llama.cpp"),
            model_format=config.get("model_format", "gguf"),
            quantization=config.get("quantization", "Q8_0"),
            gpu_layers=config.get("gpu_layers", 0),
        )
        backend = LlamaCppServerBackend()
        backend.load(llama_config)
        return backend
    raise NotImplementedError(f"Unsupported backend: {backend_type}")


def load_qwen_surface_backend(config_path: Path) -> QwenApiBackend:
    """Load a QwenApiBackend from a surface-realization inference config.

    Dependency chain:
      inference config → this factory → QwenApiBackend → StrongModelSurfaceRealizer
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("backend") != "qwen_api":
        raise ValueError(
            f"load_qwen_surface_backend requires backend=qwen_api, got {config.get('backend')}"
        )
    qwen_config = QwenApiConfig(
        model=config.get("model", ""),
        temperature=config.get("temperature", 0.8),
        max_tokens=config.get("max_tokens", 512),
        timeout_s=config.get("timeout_s", 180.0),
        api_key_env=config.get("api_key_env", "DASHSCOPE_API_KEY"),
        base_url_env=config.get("base_url_env", "DASHSCOPE_BASE_URL"),
    )
    return QwenApiBackend(qwen_config)
