"""Phase 04 Task 1 — Training config contract validator tests."""

import json
from pathlib import Path

import pytest
import yaml

from homechef_booking.training.config import (
    DatasetRegistry,
    TrainingRunSpec,
    load_training_run_spec,
    validate_training_run_spec,
)

# ── Config loading ──────────────────────────────────────────────────────────


def test_sft_config_uses_spec_defaults_for_qwen3_0_6b():
    spec = load_training_run_spec(Path("configs/training/phase04_sft_qwen3_0_6b.yaml"))
    assert spec.stage == "sft"
    assert spec.model_name_or_path == "Qwen/Qwen3-0.6B-Base"
    assert spec.finetuning_type == "lora"
    assert spec.lora_rank == 16
    assert spec.lora_alpha == 32
    assert spec.lora_dropout == 0.05
    assert spec.learning_rate == 1.0e-4
    assert spec.num_train_epochs == 3
    assert spec.cutoff_len == 2048
    assert spec.per_device_train_batch_size == 2
    assert spec.per_device_eval_batch_size == 2
    assert spec.train_dataset_path == Path("data/processed/phase03_sft_v0.1_train.jsonl")
    assert spec.eval_dataset_path == Path("data/processed/phase03_sft_v0.1_val.jsonl")
    assert spec.template == "qwen3"
    assert spec.enable_thinking is False
    assert spec.train_on_prompt is False
    assert spec.mask_history is True
    assert spec.gradient_accumulation_steps == 8
    assert spec.lr_scheduler_type == "cosine"
    assert spec.warmup_ratio == 0.1
    assert spec.bf16 is True
    assert spec.fp16 is False
    assert spec.eval_strategy == "epoch"
    assert spec.save_strategy == "epoch"
    assert spec.load_best_model_at_end is True
    assert spec.metric_for_best_model == "eval_loss"
    assert spec.greater_is_better is False
    assert spec.save_total_limit == 2
    assert spec.plot_loss is True
    assert validate_training_run_spec(spec) == []


def test_sft_config_uses_spec_defaults_for_qwen3_1_7b():
    spec = load_training_run_spec(Path("configs/training/phase04_sft_qwen3_1_7b.yaml"))
    assert spec.stage == "sft"
    assert spec.model_name_or_path == "Qwen/Qwen3-1.7B-Base"
    assert spec.finetuning_type == "lora"
    assert spec.lora_rank == 16
    assert spec.learning_rate == 1.0e-4
    assert spec.num_train_epochs == 3
    assert spec.train_dataset_path == Path("data/processed/phase03_sft_v0.1_train.jsonl")
    assert spec.eval_dataset_path == Path("data/processed/phase03_sft_v0.1_val.jsonl")
    assert spec.per_device_train_batch_size == 2
    assert spec.per_device_eval_batch_size == 2
    assert validate_training_run_spec(spec) == []


# ── Rejection of eval suites as training data ───────────────────────────────


def test_training_config_rejects_frozen_test_as_train_dataset(tmp_path: Path):
    path = tmp_path / "bad_frozen_train.yaml"
    spec_dict = {
        "stage": "sft",
        "model_name_or_path": "Qwen/Qwen3-0.6B-Base",
        "train_dataset_path": "data/eval/frozen_test.jsonl",
        "eval_dataset_path": "data/processed/phase03_sft_v0.1_val.jsonl",
    }
    path.write_text(yaml.dump(spec_dict), encoding="utf-8")
    spec = load_training_run_spec(path)
    errors = validate_training_run_spec(spec)
    assert any("Frozen Test must not be used for training" in e for e in errors)


def test_training_config_rejects_diagnostic_dev_as_train_dataset(tmp_path: Path):
    path = tmp_path / "bad_diag_train.yaml"
    spec_dict = {
        "stage": "sft",
        "model_name_or_path": "Qwen/Qwen3-0.6B-Base",
        "train_dataset_path": "data/dev/diagnostic_dev.jsonl",
        "eval_dataset_path": "data/processed/phase03_sft_v0.1_val.jsonl",
    }
    path.write_text(yaml.dump(spec_dict), encoding="utf-8")
    spec = load_training_run_spec(path)
    errors = validate_training_run_spec(spec)
    assert any("Diagnostic Dev must not be used for training" in e for e in errors)


def test_training_config_rejects_eval_suite_as_eval_dataset(tmp_path: Path):
    path = tmp_path / "bad_eval_eval.yaml"
    spec_dict = {
        "stage": "sft",
        "model_name_or_path": "Qwen/Qwen3-0.6B-Base",
        "train_dataset_path": "data/processed/phase03_sft_v0.1_train.jsonl",
        "eval_dataset_path": "data/eval/frozen_test.jsonl",
    }
    path.write_text(yaml.dump(spec_dict), encoding="utf-8")
    spec = load_training_run_spec(path)
    errors = validate_training_run_spec(spec)
    assert any("Frozen Test must not be used for training" in e for e in errors)


def test_training_config_rejects_phase02_benchmark_output_as_train_dataset(tmp_path: Path):
    path = tmp_path / "bad_phase02.yaml"
    spec_dict = {
        "stage": "sft",
        "model_name_or_path": "Qwen/Qwen3-0.6B-Base",
        "train_dataset_path": "reports/generated/phase02/mock_results.jsonl",
        "eval_dataset_path": "data/processed/phase03_sft_v0.1_val.jsonl",
    }
    path.write_text(yaml.dump(spec_dict), encoding="utf-8")
    spec = load_training_run_spec(path)
    errors = validate_training_run_spec(spec)
    assert any("Phase 02 benchmark output must not be used for training" in e for e in errors)


# ── Rejection of invalid stage values ───────────────────────────────────────


def test_training_config_rejects_unknown_stage(tmp_path: Path):
    path = tmp_path / "bad_stage.yaml"
    spec_dict = {
        "stage": "rlhf",
        "model_name_or_path": "Qwen/Qwen3-0.6B-Base",
        "train_dataset_path": "data/processed/phase03_sft_v0.1_train.jsonl",
        "eval_dataset_path": "data/processed/phase03_sft_v0.1_val.jsonl",
    }
    path.write_text(yaml.dump(spec_dict), encoding="utf-8")
    spec = load_training_run_spec(path)
    errors = validate_training_run_spec(spec)
    assert any("Unknown stage" in e for e in errors)


# ── Required field validation ───────────────────────────────────────────────


def test_training_config_rejects_missing_model(tmp_path: Path):
    path = tmp_path / "no_model.yaml"
    spec_dict = {
        "stage": "sft",
        "train_dataset_path": "data/processed/phase03_sft_v0.1_train.jsonl",
        "eval_dataset_path": "data/processed/phase03_sft_v0.1_val.jsonl",
    }
    path.write_text(yaml.dump(spec_dict), encoding="utf-8")
    spec = load_training_run_spec(path)
    errors = validate_training_run_spec(spec)
    assert any("model_name_or_path" in e for e in errors)


def test_training_config_rejects_missing_train_dataset(tmp_path: Path):
    path = tmp_path / "no_train.yaml"
    spec_dict = {
        "stage": "sft",
        "model_name_or_path": "Qwen/Qwen3-0.6B-Base",
        "eval_dataset_path": "data/processed/phase03_sft_v0.1_val.jsonl",
    }
    path.write_text(yaml.dump(spec_dict), encoding="utf-8")
    spec = load_training_run_spec(path)
    errors = validate_training_run_spec(spec)
    assert any("train_dataset_path" in e for e in errors)


# ── Batch size spec compliance ──────────────────────────────────────────────


def test_training_config_rejects_batch_size_not_2(tmp_path: Path):
    path = tmp_path / "bad_batch.yaml"
    spec_dict = {
        "stage": "sft",
        "model_name_or_path": "Qwen/Qwen3-0.6B-Base",
        "train_dataset_path": "data/processed/phase03_sft_v0.1_train.jsonl",
        "eval_dataset_path": "data/processed/phase03_sft_v0.1_val.jsonl",
        "per_device_train_batch_size": 8,
    }
    path.write_text(yaml.dump(spec_dict), encoding="utf-8")
    spec = load_training_run_spec(path)
    errors = validate_training_run_spec(spec)
    assert any("per_device_train_batch_size must be 2" in e for e in errors)


# ── Dataset registry ────────────────────────────────────────────────────────


def test_dataset_registry_loads_and_validates():
    registry = DatasetRegistry.load(Path("configs/training/phase04_dataset_info.json"))
    assert "homechef_phase03_sft_v0_1" in registry.datasets
    assert "homechef_phase03_dpo_targeted_v0_1" in registry.datasets
    sft_entry = registry.datasets["homechef_phase03_sft_v0_1"]
    assert sft_entry["formatting"] == "sharegpt"
    assert sft_entry["columns"]["messages"] == "messages"
    dpo_entry = registry.datasets["homechef_phase03_dpo_targeted_v0_1"]
    assert dpo_entry["formatting"] == "sharegpt"
    assert dpo_entry["ranking"] is True
    assert dpo_entry["columns"]["chosen"] == "chosen"
    errors = registry.validate()
    assert errors == []
