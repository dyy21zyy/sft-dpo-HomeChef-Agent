"""Phase 04 HuggingFace adapter-aware backend with true Structured Runtime.

Loads the Phase02 final base model (1.7B-Base or 4B-Instruct-2507) and
EXPLICITLY applies a trained PEFT LoRA adapter via ``adapter_name_or_path``.

Structured Output (Codex Phase04 Release Review Blocker):
- Uses LM Format Enforcer (``lm-format-enforcer``) as the HF/Transformers
  equivalent of the Phase02 llama.cpp constrained-decoding implementation.
- The JSON Schema comes from the project's CANONICAL Decision schema
  (``build_canonical_decision_json_schema``) — never copied or re-authored.
- Unstructured: ``self._model.generate(...)`` with NO schema constraint.
- Structured: ``prefix_allowed_tokens_fn`` built from
  ``JsonSchemaParser`` + ``build_transformers_prefix_allowed_tokens_fn`` is
  actually passed to ``self._model.generate(...)``.

Fail-closed guarantees:
- If the structured parser / constraint builder fails -> HARD FAIL.
- If the adapter is missing / not loadable -> HARD FAIL.
- structured failure NEVER falls back to unstructured.
- Unstructured never silently applies a constraint.

Structured and Unstructured runs use the SAME base model, SAME PEFT adapter,
SAME prompt, and SAME generation params. The ONLY business difference is
whether ``prefix_allowed_tokens_fn`` is enabled.

Provenance metadata (base_model_id / adapter_name_or_path / training_stage /
model_size / use_structured_output) is propagated into every GenerationResult
so reports can prove which trained model was evaluated (not the Base).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from homechef_booking.inference.backend import GenerationParams
from homechef_booking.inference.response import GenerationResult
from homechef_booking.inference.structured_output import (
    build_canonical_decision_json_schema,
)
from homechef_booking.prompts import Message

# Phase02 final actual model ids are the ONLY valid base-model start points.
PHASE02_FINAL_MODEL_IDS = {
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Instruct-2507",
}

# model_key -> model-size label only. training_stage is an EXPERIMENT-STAGE
# property and must NEVER be derived from model_key / model_size / model variant
# (Codex round-4 contract). It is read explicitly from the backend config.
_MODEL_KEY_TO_META = {
    "1_7b": {"model_size": "1.7B"},
    "4b": {"model_size": "4B"},
}

# Legal training_stage values for a Phase04 formal run.
_LEGAL_TRAINING_STAGES = ("sft", "dpo")


class Phase04StructuredConstraintError(RuntimeError):
    """Raised when the structured constraint cannot be built or applied.

    This is a hard, fail-closed error: a structured run must never silently
    fall back to unstructured decoding.
    """


class Phase04HFConfig(BaseModel):
    """Phase04 HF backend config with explicit adapter loading.

    ``adapter_name_or_path`` is REQUIRED for a Phase04 run — a run without an
    adapter would be a Base evaluation, which is not part of Phase04 formal.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    model_id: str
    adapter_name_or_path: str | None = None
    device: str = "auto"
    torch_dtype: str = "float32"
    max_model_length: int = 2048
    max_new_tokens: int = 512
    timeout_seconds: float | None = 120.0
    use_structured_output: bool = False
    # Provenance metadata (recorded into every GenerationResult).
    model_key: str | None = None
    # training_stage is REQUIRED and must be explicitly provided by the backend
    # config ("sft" | "dpo"). It is an experiment-stage property and must NOT be
    # derived from model_key / model_size / model variant. Missing/invalid ->
    # fail-closed.
    training_stage: str | None = None
    model_size: str | None = None
    # pref_beta: None for SFT; 0.1/0.3 for DPO (beta sweep provenance).
    pref_beta: float | None = None
    loaded_adapter: str | None = Field(default=None, exclude=False)

    def validate_for_run(self) -> list[str]:
        errors: list[str] = []
        if self.model_id not in PHASE02_FINAL_MODEL_IDS:
            errors.append(
                f"Phase04 base model '{self.model_id}' is not a Phase02 final model. "
                f"Allowed: {sorted(PHASE02_FINAL_MODEL_IDS)}."
            )
        if not self.adapter_name_or_path:
            errors.append(
                "adapter_name_or_path is required for a Phase04 run (non-Base proof). "
                "A Phase04 run must evaluate the trained SFT/DPO adapter, not the Base."
            )
        # training_stage must be explicit and legal (fail-closed, no default sft).
        if not self.training_stage:
            errors.append(
                "training_stage is required for a Phase04 run (experiment stage, "
                "not derivable from model identity). Provide 'sft' or 'dpo'."
            )
        elif self.training_stage not in _LEGAL_TRAINING_STAGES:
            errors.append(
                f"training_stage '{self.training_stage}' is invalid. "
                f"Legal values: {sorted(_LEGAL_TRAINING_STAGES)}."
            )
        return errors


class Phase04HFTransformersBackend:
    name = "phase04_hf"

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._config: Phase04HFConfig | None = None
        self._constraint_fn: Callable | None = None

    # ── constraint builder (pure, unit-testable) ───────────────────────────

    @staticmethod
    def build_constraint_fn(tokenizer) -> Callable:
        """Build a transformers ``prefix_allowed_tokens_fn`` from the canonical
        Decision JSON Schema via LM Format Enforcer.

        Imports are lazy so the module imports cleanly without LFE installed;
        tests inject fakes via ``_build_constraint_fn`` monkeypatching.
        """
        try:
            # lm-format-enforcer ships its top-level package as "lmformatenforcer".
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import (
                build_transformers_prefix_allowed_tokens_fn,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            raise Phase04StructuredConstraintError(
                "lm-format-enforcer is required for Phase04 structured output: "
                f"{exc}"
            ) from exc

        # Canonical schema source — never copied / re-authored.
        schema = build_canonical_decision_json_schema()
        try:
            parser = JsonSchemaParser(schema)
        except Exception as exc:
            raise Phase04StructuredConstraintError(
                f"Failed to compile canonical Decision JSON Schema: {exc}"
            ) from exc
        try:
            return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
        except Exception as exc:
            raise Phase04StructuredConstraintError(
                f"Failed to build prefix_allowed_tokens_fn: {exc}"
            ) from exc

    # ── adapter marker (fail-closed) ────────────────────────────────────────

    @staticmethod
    def _require_adapter_marker(adapter: Path) -> Path:
        """Fail-closed: ensure a real LLaMA-Factory adapter dir exists.

        Raises ValueError (hard fail) if the adapter_config.json marker is
        missing. Never falls back to Base weights.
        """
        if not (adapter / "adapter_config.json").exists():
            raise ValueError(
                f"Adapter not found at {adapter} (no adapter_config.json). "
                "Phase04 runs must use a trained SFT/DPO adapter; refusing Base fallback."
            )
        return adapter

    # ── load ────────────────────────────────────────────────────────────────

    def load(self, config: Phase04HFConfig) -> None:
        errors = config.validate_for_run()
        if errors:
            raise ValueError("; ".join(errors))

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        torch_dtype = getattr(torch, config.torch_dtype, torch.float32)

        self._tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
        if config.device == "cpu":
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        elif config.device == "auto":
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                torch_dtype=torch_dtype,
                device_map={"": config.device},
                trust_remote_code=True,
            )

        # Fail-closed: refuse to evaluate Base-only weights.
        adapter = self._require_adapter_marker(Path(config.adapter_name_or_path))
        self._model = PeftModel.from_pretrained(base_model, adapter)
        # Non-Base proof: record the resolved adapter.
        config.loaded_adapter = str(adapter)
        self._config = config

        # Structured runtime: build the constraint fn at load time.
        if config.use_structured_output:
            self._constraint_fn = self.build_constraint_fn(self._tokenizer)
        else:
            self._constraint_fn = None

    # ── provenance helpers ──────────────────────────────────────────────────

    def _provenance_kwargs(self) -> dict:
        """Build provenance metadata from the explicit config.

        training_stage is read ONLY from config.training_stage (fail-closed,
        no model-key/size/variant derivation). model_size may use the
        model-key fallback because it IS a model-identity property.
        """
        cfg = self._config
        if cfg is None:
            return {}
        training_stage = cfg.training_stage
        model_size = cfg.model_size
        if cfg.model_key and cfg.model_key in _MODEL_KEY_TO_META:
            model_size = model_size or _MODEL_KEY_TO_META[cfg.model_key]["model_size"]
        return {
            "base_model_id": cfg.model_id,
            "adapter_name_or_path": cfg.loaded_adapter or cfg.adapter_name_or_path,
            "training_stage": training_stage,
            "model_size": model_size,
            "use_structured_output": cfg.use_structured_output,
            "pref_beta": cfg.pref_beta,
        }

    # ── generate ────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_messages_for_chat_template(messages):
        """Normalize tool-call history for HF chat-template compatibility.

        Preserve tool_calls exactly. Only convert assistant tool-call
        messages with content=None to content="".
        """
        normalized = []

        for message in messages:
            item = dict(message)

            if (
                item.get("role") == "assistant"
                and item.get("tool_calls")
                and item.get("content") is None
            ):
                item["content"] = ""

            normalized.append(item)

        return normalized

    def generate(
        self,
        messages: list[Message],
        params: GenerationParams,
        case_id: str | None = None,
    ) -> GenerationResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Phase04HFTransformersBackend not loaded. Call load(config) first.")
        import torch

        selected = case_id or "unknown"
        start = time.perf_counter()
        provenance = self._provenance_kwargs()
        structured = bool(self._config and self._config.use_structured_output)

        try:
            template_messages = self._normalize_messages_for_chat_template(messages)

            text = self._tokenizer.apply_chat_template(
                template_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text, return_tensors="pt")
            if self._config and self._config.device != "cpu":
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            with torch.no_grad():
                if structured:
                    # Structured: MUST pass the schema constraint.
                    if self._constraint_fn is None:
                        raise Phase04StructuredConstraintError(
                            "Structured run has no prefix_allowed_tokens_fn; "
                            "refusing to generate unstructured."
                        )
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=self._config.max_new_tokens if self._config else 512,
                        do_sample=False,
                        prefix_allowed_tokens_fn=self._constraint_fn,
                    )
                else:
                    # Unstructured: MUST NOT pass any schema constraint.
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=self._config.max_new_tokens if self._config else 512,
                        do_sample=False,
                    )
            raw_text = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            elapsed = (time.perf_counter() - start) * 1000
            return GenerationResult(
                case_id=selected,
                backend_name=self.name,
                raw_text=raw_text,
                finish_reason="stop",
                latency_ms=round(elapsed, 2),
                **provenance,
            )
        except Phase04StructuredConstraintError:
            # Hard, fail-closed. Never fall back to unstructured.
            elapsed = (time.perf_counter() - start) * 1000
            return GenerationResult(
                case_id=selected,
                backend_name=self.name,
                error_type="structured_constraint_error",
                error_message="Structured constraint failure (fail-closed); no unstructured fallback.",
                latency_ms=round(elapsed, 2),
                **provenance,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return GenerationResult(
                case_id=selected,
                backend_name=self.name,
                error_type=type(exc).__name__,
                error_message=str(exc),
                latency_ms=round(elapsed, 2),
                **provenance,
            )
