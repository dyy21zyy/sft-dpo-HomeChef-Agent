"""Phase 04 formal training-config renderer contract.

Shared by the formal renderer (scripts/train/render_config.py) and the dry-run
renderer (scripts/train/dryrun.py):

- ``LLAMAFACTORY_ALLOWED_ARGS``: the systematic allowlist of LLaMA-Factory
  native args. Project-only metadata (experiment_class, train_dataset_path,
  eval_dataset_path, engineering_dryrun_only, sample_count, approval_required)
  is NEVER forwarded.
- ``filter_llamafactory_args``: keep only allowlisted keys.
- ``resolve_dataset_registry_name``: map a project ``data/...`` path to the
  LLaMA-Factory ``data/dataset_info.json`` registry name (fail-closed).
- ``assert_rendered_config``: fail-closed gate before launching LLaMA-Factory
  (do_train=true, dataset/eval_dataset/model/stage present, no project fields).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# LLaMA-Factory 0.9.5 native argument allowlist (shared contract).
LLAMAFACTORY_ALLOWED_ARGS: frozenset[str] = frozenset({
    "stage",
    "do_train",
    "do_eval",
    "model_name_or_path",
    "finetuning_type",
    "lora_target",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "template",
    "enable_thinking",
    "train_on_prompt",
    "mask_history",
    "cutoff_len",
    "learning_rate",
    "num_train_epochs",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "lr_scheduler_type",
    "warmup_ratio",
    "bf16",
    "fp16",
    "eval_strategy",
    "save_strategy",
    "save_total_limit",
    "load_best_model_at_end",
    "metric_for_best_model",
    "greater_is_better",
    "plot_loss",
    "adapter_name_or_path",
    "pref_beta",
    "pref_loss",
    "pref_ftx",
    "dpo_loss",
    "dpo_ftx",
    "output_dir",
    "max_steps",
    "dataset",
    "eval_dataset",
    "use_cpu",
})

# Project-only metadata that must NEVER reach LLaMA-Factory.
PROJECT_ONLY_FIELDS: frozenset[str] = frozenset({
    "train_dataset_path",
    "eval_dataset_path",
    "experiment_class",
    "engineering_dryrun_only",
    "sample_count",
    "approval_required",
})


class Phase04TrainingConfigError(RuntimeError):
    """Raised when a rendered training config is invalid / not actually trainable."""


def filter_llamafactory_args(raw: dict[str, Any]) -> dict[str, Any]:
    """Return only the LLaMA-Factory native args from ``raw``.

    Drops project-only metadata (experiment_class, train_dataset_path, etc.).
    """
    return {k: v for k, v in raw.items() if k in LLAMAFACTORY_ALLOWED_ARGS}


def _normalize_registry_path(p: str | Path) -> str:
    """Normalize a path for registry comparison (POSIX, strip a leading data/).

    project config: "data/processed/sft/v0.3/train.jsonl"
    registry file:  "processed/sft/v0.3/train.jsonl"
    """
    s = str(p).replace("\\", "/")
    # Strip a leading "data/" prefix so both forms normalize the same.
    for prefix in ("data/", "./data/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s


def resolve_dataset_registry_name(
    dataset_path: str | Path,
    dataset_info_path: Path = Path("data/dataset_info.json"),
) -> str:
    """Map a project ``data/...`` path to the LLaMA-Factory registry name.

    Reads data/dataset_info.json, normalizes paths (handles the data/ prefix),
    and returns the SINGLE matching registry name. 0 or >1 matches -> hard fail
    (no silent fallback, no registry mutation, no name guessing, no v0.1/0.6B).
    """
    if not dataset_info_path.exists():
        raise Phase04TrainingConfigError(
            f"Dataset registry not found: {dataset_info_path}"
        )
    registry = yaml.safe_load(dataset_info_path.read_text(encoding="utf-8")) or {}
    target = _normalize_registry_path(dataset_path)

    matches = [
        name
        for name, entry in registry.items()
        if isinstance(entry, dict) and _normalize_registry_path(entry.get("file_name", "")) == target
    ]
    if not matches:
        raise Phase04TrainingConfigError(
            f"No dataset registry entry matches dataset path: {dataset_path!r} "
            f"(normalized {target!r}). No fallback allowed."
        )
    if len(matches) > 1:
        raise Phase04TrainingConfigError(
            f"Multiple dataset registry entries match {dataset_path!r}: {matches}. "
            "Ambiguous; refusing to guess."
        )
    return matches[0]


def render_dataset_fields(raw: dict[str, Any], dataset_info_path: Path = Path("data/dataset_info.json")) -> None:
    """In-place: convert project dataset paths to LLaMA-Factory registry names.

    - raw["train_dataset_path"]  -> raw["dataset"]
    - raw["eval_dataset_path"]   -> raw["eval_dataset"]
    then deletes the project path fields.
    """
    if raw.get("train_dataset_path"):
        raw["dataset"] = resolve_dataset_registry_name(raw["train_dataset_path"], dataset_info_path)
        raw.pop("train_dataset_path", None)
    if raw.get("eval_dataset_path"):
        raw["eval_dataset"] = resolve_dataset_registry_name(raw["eval_dataset_path"], dataset_info_path)
        raw.pop("eval_dataset_path", None)


def assert_rendered_config(resolved: dict[str, Any], *, stage: str | None = None) -> bool:
    """Fail-closed gate on a resolved LLaMA-Factory training config.

    Requires: do_train=true, dataset, eval_dataset, model_name_or_path, stage.
    For DPO: adapter_name_or_path and pref_beta present.
    Forbids any leaked project-only metadata.
    """
    _stage = resolved.get("stage", stage)
    if _stage not in ("sft", "dpo"):
        raise Phase04TrainingConfigError(
            f"Rendered config must have stage sft|dpo, got {_stage!r}."
        )
    if resolved.get("do_train") is not True:
        raise Phase04TrainingConfigError(
            "Rendered training config must set do_train=true "
            "(LLaMA-Factory otherwise skips the training branch)."
        )
    if not resolved.get("dataset"):
        raise Phase04TrainingConfigError("Rendered config missing dataset (registry name).")
    if not resolved.get("eval_dataset"):
        raise Phase04TrainingConfigError("Rendered config missing eval_dataset (registry name).")
    if not resolved.get("model_name_or_path"):
        raise Phase04TrainingConfigError("Rendered config missing model_name_or_path.")
    if _stage == "dpo":
        if not resolved.get("adapter_name_or_path"):
            raise Phase04TrainingConfigError(
                "DPO rendered config missing adapter_name_or_path (Base->SFT->DPO)."
            )
        if resolved.get("pref_beta") is None:
            raise Phase04TrainingConfigError("DPO rendered config missing pref_beta.")
    # No project-only metadata may leak.
    leaked = [f for f in PROJECT_ONLY_FIELDS if f in resolved]
    if leaked:
        raise Phase04TrainingConfigError(
            f"Rendered config leaked project-only metadata: {leaked}"
        )
    return True


def render_training_spec(
    spec,
    dataset_info_path: Path = Path("data/dataset_info.json"),
    *,
    dryrun: bool = False,
) -> dict[str, Any]:
    """Render a TrainingRunSpec into a LLaMA-Factory-native config dict.

    - Filters to allowlisted args.
    - Converts dataset paths to registry names.
    - dryrun=True forces do_train=true + do_eval=false + CPU overrides.
    """
    raw = spec.model_dump(mode="json", exclude_none=True)
    # Convert dataset paths to registry names BEFORE allowlist filtering
    # (train_dataset_path/eval_dataset_path are project-only and would be dropped).
    render_dataset_fields(raw, dataset_info_path)
    resolved = filter_llamafactory_args(raw)
    # mask_history / train_on_prompt are SFT-only per LLaMA-Factory 0.9.5 parser;
    # they must NOT appear in a DPO config (even the Pydantic default mask_history=True).
    if spec.stage != "sft":
        resolved.pop("mask_history", None)
        resolved.pop("train_on_prompt", None)
    if dryrun:
        resolved["do_train"] = True
        resolved["do_eval"] = False
    return resolved
