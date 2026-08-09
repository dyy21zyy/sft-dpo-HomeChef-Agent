"""Task 4 Phase 02: HuggingFace Transformers backend tests."""

from pathlib import Path

import pytest

from homechef_booking.inference.hf_backend import HFTransformersBackend, HFBackendConfig
from homechef_booking.inference.backend import GenerationParams


def test_hf_backend_config_accepts_valid_model_ids():
    config = HFBackendConfig(model_id="Qwen/Qwen3-0.6B-Base", device="cpu", torch_dtype="float32", max_model_length=2048)
    assert config.model_id == "Qwen/Qwen3-0.6B-Base"
    assert config.device == "cpu"


def test_hf_backend_config_rejects_sft_model():
    with pytest.raises(ValueError, match="sft"):
        HFBackendConfig(model_id="Qwen/Qwen3-0.6B-Instruct")


def test_hf_backend_config_rejects_dpo_model():
    with pytest.raises(ValueError, match="dpo"):
        HFBackendConfig(model_id="Qwen/Qwen3-0.6B-DPO")


def test_hf_backend_config_rejects_quantized_model():
    with pytest.raises(ValueError, match="quant"):
        HFBackendConfig(model_id="Qwen/Qwen3-0.6B-GGUF")


def test_hf_backend_config_rejects_lora_adapter():
    with pytest.raises(ValueError, match="lora"):
        HFBackendConfig(model_id="Qwen/Qwen3-0.6B-LoRA")


def test_hf_backend_config_accepts_qwen3_1_7b_base():
    config = HFBackendConfig(model_id="Qwen/Qwen3-1.7B-Base", device="cpu", torch_dtype="float32", max_model_length=2048)
    assert config.model_id == "Qwen/Qwen3-1.7B-Base"


def test_hf_backend_config_has_default_device():
    config = HFBackendConfig(model_id="Qwen/Qwen3-0.6B-Base")
    assert config.device == "auto"


def test_hf_backend_generate_raises_not_loaded_error():
    backend = HFTransformersBackend()
    with pytest.raises(RuntimeError, match="not loaded"):
        backend.generate([{"role": "user", "content": "你好"}], GenerationParams())


def test_hf_backend_name_is_hf_transformers():
    backend = HFTransformersBackend()
    assert backend.name == "hf_transformers"
