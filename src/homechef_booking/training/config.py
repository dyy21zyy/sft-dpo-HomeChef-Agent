"""Phase 04 Training Run Spec — strict Pydantic models and fail-closed validation.

Loads SFT/DPO training YAML configs, validates SPEC defaults, and rejects
eval suites (Frozen Test, Diagnostic Dev) and Phase 02 benchmark outputs
as training or evaluation data.
"""

import json
from pathlib import Path
from typing import Any, Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, BeforeValidator


def _coerce_stage(v: Any) -> str:
    """Coerce stage to a known string."""
    if isinstance(v, str):
        return v
    return str(v)


def _coerce_path(v: Any) -> Path | None:
    """Coerce a string or Path to Path, or return None."""
    if v is None:
        return None
    return Path(str(v))


_VALID_STAGES = {"sft", "dpo"}

# Frozen Test and Diagnostic Dev must not be used
_EVAL_ONLY_PATHS = {
    "data/eval/frozen_test.jsonl": "Frozen Test must not be used for training",
    "data/dev/diagnostic_dev.jsonl": "Diagnostic Dev must not be used for training",
}

_FORBIDDEN_PHASE02_OUTPUT_PATHS = [
    "reports/generated/phase02",
]


class TrainingRunSpec(BaseModel):
    """Strict training run configuration loaded from YAML.

    SFT defaults follow the SPEC-approved values from the Phase 04 frozen plan.
    DPO configs will be added after SFT approval.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    stage: str = Field(default="sft")
    model_name_or_path: str = Field(default="Qwen/Qwen3-0.6B-Base")

    # ── LoRA ──
    finetuning_type: str = Field(default="lora")
    lora_target: str = Field(default="all")
    lora_rank: int = Field(default=16)
    lora_alpha: int = Field(default=32)
    lora_dropout: float = Field(default=0.05)

    # ── Template ──
    template: str = Field(default="qwen3")
    enable_thinking: bool = Field(default=False)
    train_on_prompt: bool = Field(default=False)
    mask_history: bool = Field(default=True)

    # ── Sequence ──
    cutoff_len: int = Field(default=2048)

    # ── Optimization ──
    learning_rate: float = Field(default=1.0e-4)
    num_train_epochs: int = Field(default=3)
    per_device_train_batch_size: int = Field(default=2)
    per_device_eval_batch_size: int = Field(default=2)
    gradient_accumulation_steps: int = Field(default=8)
    lr_scheduler_type: str = Field(default="cosine")
    warmup_ratio: float = Field(default=0.1)

    # ── Precision ──
    bf16: bool = Field(default=True)
    fp16: bool = Field(default=False)

    # ── Checkpointing ──
    eval_strategy: str = Field(default="epoch")
    save_strategy: str = Field(default="epoch")
    load_best_model_at_end: bool = Field(default=True)
    metric_for_best_model: str = Field(default="eval_loss")
    greater_is_better: bool = Field(default=False)
    save_total_limit: int = Field(default=2)
    plot_loss: bool = Field(default=True)

    # ── Datasets (coerced from str to Path) ──
    train_dataset_path: Path | None = Field(default=None)
    eval_dataset_path: Path | None = Field(default=None)

    # ── DPO-specific (not validated yet — requires SFT approval) ──
    adapter_name_or_path: Path | None = Field(default=None)
    pref_beta: float | None = Field(default=None)
    dpo_loss: str | None = Field(default=None)
    dpo_ftx: float | None = Field(default=None)

    # ── Dry-run only (engineering validation, not for formal training) ──
    engineering_dryrun_only: bool = Field(default=False)
    max_steps: int | None = Field(default=None)
    sample_count: int | None = Field(default=None)

    # ── Output ──
    output_dir: Path | None = Field(default=None)

    # ── Approval metadata (informational, not validated) ──
    approval_required: dict[str, str] | None = Field(default=None)

    @classmethod
    def _validate_path_coercion(cls, data: dict) -> dict:
        """Coerce string paths to Path objects before model init."""
        for key in ("train_dataset_path", "eval_dataset_path", "adapter_name_or_path", "output_dir"):
            if key in data and data[key] is not None and isinstance(data[key], str):
                data[key] = Path(data[key])
        return data


def _is_forbidden_path(p: Path) -> list[str]:
    """Check if a path is forbidden for training. Returns list of error messages."""
    errors = []
    path_str = str(p)
    # Check eval-only paths
    for forbidden_path, msg in _EVAL_ONLY_PATHS.items():
        if forbidden_path in path_str:
            errors.append(msg)
    # Check Phase 02 output paths
    for forbidden in _FORBIDDEN_PHASE02_OUTPUT_PATHS:
        if forbidden in path_str:
            errors.append(f"Phase 02 benchmark output must not be used for training: {path_str}")
    return errors


def load_training_run_spec(path: Path) -> TrainingRunSpec:
    """Load and validate a training run spec from YAML.

    All SPEC defaults are applied via Pydantic field defaults; the YAML file
    may override any field. Unrecognized fields are rejected (extra="forbid").
    Path fields are coerced from str to Path.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    # Coerce string paths to Path objects
    raw = TrainingRunSpec._validate_path_coercion(raw)
    return TrainingRunSpec(**raw)


def validate_training_run_spec(spec: TrainingRunSpec) -> list[str]:
    """Run fail-closed validation on a TrainingRunSpec.

    Returns a list of error messages. An empty list means the spec is valid.
    """
    errors = []

    # Stage must be known
    if spec.stage not in _VALID_STAGES:
        errors.append(f"Unknown stage: {spec.stage}. Must be 'sft' or 'dpo'.")

    # model_name_or_path is required
    if not spec.model_name_or_path:
        errors.append("model_name_or_path is required.")

    # train_dataset_path is required
    if spec.train_dataset_path is None:
        errors.append("train_dataset_path is required.")
    else:
        errors.extend(_is_forbidden_path(spec.train_dataset_path))
        if not spec.train_dataset_path.exists():
            errors.append(f"train_dataset_path does not exist: {spec.train_dataset_path}")

    # eval_dataset_path is required
    if spec.eval_dataset_path is None:
        errors.append("eval_dataset_path is required.")
    else:
        errors.extend(_is_forbidden_path(spec.eval_dataset_path))
        if not spec.eval_dataset_path.exists():
            errors.append(f"eval_dataset_path does not exist: {spec.eval_dataset_path}")

    # Batch size must be 2
    if spec.per_device_train_batch_size != 2:
        errors.append(f"per_device_train_batch_size must be 2, got {spec.per_device_train_batch_size}.")
    if spec.per_device_eval_batch_size != 2:
        errors.append(f"per_device_eval_batch_size must be 2, got {spec.per_device_eval_batch_size}.")

    return errors


class DatasetRegistry(BaseModel):
    """LLaMA-Factory dataset registry (dataset_info.json).

    Maps dataset names to LLaMA-Factory format specifications.
    """

    model_config = ConfigDict(extra="allow", strict=False)

    datasets: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "DatasetRegistry":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(datasets=raw)

    def validate(self) -> list[str]:
        errors = []
        for name, entry in self.datasets.items():
            if "file_name" not in entry:
                errors.append(f"Dataset '{name}' missing 'file_name'.")
            else:
                file_path = Path(entry["file_name"])
                if not file_path.exists():
                    errors.append(f"Dataset '{name}' file not found: {entry['file_name']}")
            if "formatting" not in entry:
                errors.append(f"Dataset '{name}' missing 'formatting'.")
            if "columns" not in entry:
                errors.append(f"Dataset '{name}' missing 'columns'.")
        return errors
