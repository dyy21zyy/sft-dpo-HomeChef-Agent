from __future__ import annotations

import json
from pathlib import Path

import yaml

from homechef_booking.inference.backend import Backend
from homechef_booking.inference.hf_backend import HFBackendConfig, HFTransformersBackend
from homechef_booking.inference.mock_backend import MockBackend


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
    raise NotImplementedError(f"Unsupported backend: {backend_type}")
