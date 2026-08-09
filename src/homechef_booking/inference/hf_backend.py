"""Phase 02 HuggingFace Transformers local backend.

Supports only: Qwen/Qwen3-0.6B-Base and Qwen/Qwen3-1.7B-Base.
Explicitly rejects SFT/DPO/Instruct/GGUF/LoRA/quantized model variants.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

from homechef_booking.inference.backend import GenerationParams
from homechef_booking.inference.response import GenerationResult
from homechef_booking.prompts import Message

ALLOWED_MODEL_IDS = {
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base",
}

FORBIDDEN_MODEL_PATTERNS = [
    (r"(?i)\b(instruct|chat)\b", "sft"),
    (r"(?i)\b(dpo|rlhf|rm|reward)\b", "dpo"),
    (r"(?i)\b(gguf|ggml|awq|gptq|bnb|4bit|8bit|int4|int8|nf4|quant)\b", "quant"),
    (r"(?i)\b(lora|qlora|peft|adapter)\b", "lora"),
]


class HFBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model_id: str
    device: str = "auto"
    torch_dtype: str = "float32"
    max_model_length: int = 2048
    max_new_tokens: int = 512
    timeout_seconds: float | None = 120.0

    @model_validator(mode="after")
    def _validate_model_id(self) -> "HFBackendConfig":
        if self.model_id in ALLOWED_MODEL_IDS:
            return self
        for pattern, category in FORBIDDEN_MODEL_PATTERNS:
            if re.search(pattern, self.model_id):
                raise ValueError(f"Phase 02 does not support {category} models: {self.model_id}")
        raise ValueError(f"Phase 02 only supports Qwen/Qwen3-0.6B-Base and Qwen/Qwen3-1.7B-Base. Got: {self.model_id}")


class HFTransformersBackend:
    name = "hf_transformers"

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def load(self, config: HFBackendConfig) -> None:
        self._config = config
        self._loaded = False

    def generate(self, messages: list[Message], params: GenerationParams, case_id: str | None = None) -> GenerationResult:
        if not hasattr(self, "_config"):
            raise RuntimeError("HFTransformersBackend not loaded. Call load(config) first.")
        return GenerationResult(case_id=case_id or "unknown", backend_name=self.name, error_type="not_implemented", error_message="Real model inference not implemented in Phase 02 scope.")
