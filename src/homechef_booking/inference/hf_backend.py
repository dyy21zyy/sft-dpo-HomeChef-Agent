"""Phase 02 HuggingFace Transformers local backend.

Supports only: Qwen/Qwen3-0.6B-Base and Qwen/Qwen3-1.7B-Base.
Explicitly rejects SFT/DPO/Instruct/GGUF/LoRA/quantized model variants.
"""

from __future__ import annotations

import os
import re
import time

from pydantic import BaseModel, ConfigDict, model_validator

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
    def _validate_model_id(self) -> HFBackendConfig:
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
        self._config: HFBackendConfig | None = None

    def load(self, config: HFBackendConfig) -> None:
        self._config = config
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        torch_dtype = getattr(torch, config.torch_dtype, torch.float32)
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
        if config.device in ("cpu", "auto"):
            self._model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                torch_dtype=torch_dtype,
                device_map=config.device,
                trust_remote_code=True,
            )

    def generate(self, messages: list[Message], params: GenerationParams, case_id: str | None = None) -> GenerationResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("HFTransformersBackend not loaded. Call load(config) first.")
        import torch
        selected = case_id or "unknown"
        start = time.perf_counter()
        try:
            text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._tokenizer(text, return_tensors="pt")
            if self._config and self._config.device != "cpu":
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.generate(**inputs, max_new_tokens=self._config.max_new_tokens if self._config else 512, do_sample=False)
            raw_text = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            elapsed = (time.perf_counter() - start) * 1000
            return GenerationResult(case_id=selected, backend_name=self.name, raw_text=raw_text, finish_reason="stop", latency_ms=round(elapsed, 2))
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return GenerationResult(case_id=selected, backend_name=self.name, error_type=type(exc).__name__, error_message=str(exc), latency_ms=round(elapsed, 2))
