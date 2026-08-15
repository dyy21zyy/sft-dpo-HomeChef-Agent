"""Phase 04 DPO beta sweep + CPU dry-run + renderer TDD tests.

Covers the current user's highest-priority decisions:
- Formal models = {1.7B, 4B} only (0.6B historical/non-formal).
- 4 formal DPO configs: beta 0.1/0.3 x 1.7B/4B; control params identical.
- DPO continues from SFT adapter (Base->SFT->DPO), missing adapter hard-fails.
- CPU engineering dry-run uses Qwen/Qwen3-1.7B-Base (offline), no GPU.
- Renderer forwards ONLY LLaMA-Factory native args (allowlist).
- Dry-run PASS requires real adapter artifacts.
- 12-run eval matrix: SFT=4, DPO beta=.1=4, DPO beta=.3=4.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import scripts.train.dryrun as dryrun_mod
from homechef_booking.training.config import load_training_run_spec
from homechef_booking.training.formal_matrix import (
    FORMAL_MODELS,
    build_phase04_eval_matrix,
    validate_dpo_source_checkpoint,
)

TRAINING = Path("configs/training")

DPO_BETA_CONFIGS = [
    "phase04_dpo_qwen3_1_7b_beta_0_1.yaml",
    "phase04_dpo_qwen3_1_7b_beta_0_3.yaml",
    "phase04_dpo_qwen3_4b_beta_0_1.yaml",
    "phase04_dpo_qwen3_4b_beta_0_3.yaml",
]

# Control params that MUST be identical across beta 0.1/0.3 for a given model.
_CONTROL_KEYS = [
    "stage", "model_name_or_path", "finetuning_type", "lora_target", "lora_rank",
    "lora_alpha", "lora_dropout", "pref_loss", "learning_rate", "num_train_epochs",
    "per_device_train_batch_size", "per_device_eval_batch_size",
    "gradient_accumulation_steps", "cutoff_len", "lr_scheduler_type", "warmup_ratio",
    "bf16", "fp16", "template", "adapter_name_or_path",
]


# ── A. Formal scope: FORMAL_MODELS == {1.7B, 4B}, no 0.6B ────────────────────


def test_formal_models_only_1_7b_and_4b():
    assert set(FORMAL_MODELS) == {"1_7b", "4b"}
    ids = {m["model_id"] for m in FORMAL_MODELS.values()}
    assert ids == {"Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Instruct-2507"}
    assert "Qwen/Qwen3-0.6B-Base" not in ids


def test_formal_matrix_has_no_0_6b():
    runs = build_phase04_eval_matrix()
    for r in runs:
        assert "0.6B" not in r.model_id


# ── B. Beta configs ───────────────────────────────────────────────────────────


def test_four_formal_dpo_configs_exist():
    for c in DPO_BETA_CONFIGS:
        assert (TRAINING / c).exists(), f"missing {c}"


def test_beta_sets_are_0_1_and_0_3():
    for model in ("1_7b", "4b"):
        betas = {
            load_training_run_spec(TRAINING / f"phase04_dpo_qwen3_{model}_beta_{b}.yaml").pref_beta
            for b in ("0_1", "0_3")
        }
        assert betas == {0.1, 0.3}, f"model {model} beta set must be {{0.1, 0.3}}"


def test_beta_01_03_control_params_identical():
    # beta .1/.3 for a given model must be identical on all CONTROL_KEYS.
    for model in ("1_7b", "4b"):
        b01 = load_training_run_spec(TRAINING / f"phase04_dpo_qwen3_{model}_beta_0_1.yaml")
        b03 = load_training_run_spec(TRAINING / f"phase04_dpo_qwen3_{model}_beta_0_3.yaml")
        for key in _CONTROL_KEYS:
            assert getattr(b01, key) == getattr(b03, key), (
                f"model {model} key {key} differs across beta: {getattr(b01,key)} vs {getattr(b03,key)}"
            )
        # Only pref_beta differs (plus output_dir).
        assert b01.pref_beta != b03.pref_beta


def test_dpo_batch_is_1_1():
    for c in DPO_BETA_CONFIGS:
        spec = load_training_run_spec(TRAINING / c)
        assert spec.per_device_train_batch_size == 1
        assert spec.per_device_eval_batch_size == 1


# ── C. Model identity ─────────────────────────────────────────────────────────


def test_model_identity_1_7b_base_4b_instruct():
    for c in DPO_BETA_CONFIGS:
        spec = load_training_run_spec(TRAINING / c)
        if "1_7b" in c:
            assert spec.model_name_or_path == "Qwen/Qwen3-1.7B-Base"
        else:
            assert spec.model_name_or_path == "Qwen/Qwen3-4B-Instruct-2507"
            assert spec.model_name_or_path != "Qwen/Qwen3-4B-Base"


def test_4b_template_is_qwen3_nothink():
    for c in ("phase04_dpo_qwen3_4b_beta_0_1.yaml", "phase04_dpo_qwen3_4b_beta_0_3.yaml"):
        spec = load_training_run_spec(TRAINING / c)
        assert spec.template == "qwen3_nothink"


def test_1_7b_template_is_qwen3():
    for c in ("phase04_dpo_qwen3_1_7b_beta_0_1.yaml", "phase04_dpo_qwen3_1_7b_beta_0_3.yaml"):
        spec = load_training_run_spec(TRAINING / c)
        assert spec.template == "qwen3"


# ── D. DPO chain (Base->SFT->DPO, missing adapter hard-fail) ─────────────────


def test_dpo_references_sft_adapter():
    for c in DPO_BETA_CONFIGS:
        spec = load_training_run_spec(TRAINING / c)
        assert spec.adapter_name_or_path is not None
        # DPO output dir is beta-specific; SFT source is the shared SFT checkpoint.
        assert "checkpoint-best" in str(spec.adapter_name_or_path)
        assert "sft" in str(spec.adapter_name_or_path).lower()


def test_dpo_missing_adapter_hard_fail(tmp_path: Path):
    path = tmp_path / "dpo_no_adapter.yaml"
    path.write_text(yaml.dump({
        "stage": "dpo",
        "experiment_class": "formal",
        "model_name_or_path": "Qwen/Qwen3-1.7B-Base",
        "train_dataset_path": "data/processed/dpo/v0.3/train.jsonl",
        "eval_dataset_path": "data/processed/dpo/v0.3/val.jsonl",
    }), encoding="utf-8")
    spec = load_training_run_spec(path)
    errors = validate_dpo_source_checkpoint(spec)
    assert any("adapter_name_or_path" in e for e in errors)
    assert any("must NOT start from Base" in e for e in errors)


# ── E. CPU dry-run forced params + local model override ──────────────────────


def _render_sft_config(tmp_path: Path) -> Path:
    src = TRAINING / "phase04_sft_qwen3_1_7b.yaml"
    out = tmp_path / "rendered_sft.yaml"
    dryrun_mod._generate_dryrun_config(
        src, out, Path("experiments/phase04/dryrun/sft_1_7b"), "sft",
    )
    return out


def test_rendered_cpu_params_forced(tmp_path: Path):
    out = _render_sft_config(tmp_path)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["use_cpu"] is True
    assert data["bf16"] is False
    assert data["fp16"] is False
    assert data["per_device_train_batch_size"] == 1
    assert data["per_device_eval_batch_size"] == 1
    assert data["num_train_epochs"] == 1
    assert data["max_steps"] == 2


def test_local_model_path_override(tmp_path: Path):
    src = TRAINING / "phase04_sft_qwen3_1_7b.yaml"
    out = tmp_path / "rendered_override.yaml"
    dryrun_mod._generate_dryrun_config(
        src, out, Path("experiments/phase04/dryrun/sft_1_7b"), "sft",
        local_model_path=Path("models/hf_cache/Qwen_Qwen3-1.7B-Base"),
    )
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    # Rendered config uses resolved local path; canonical id retained separately.
    rendered = str(data["model_name_or_path"]).replace("\\", "/")
    assert "models/hf_cache/Qwen_Qwen3-1.7B-Base" in rendered


def test_missing_local_model_hard_fail(tmp_path: Path):
    import pytest
    with pytest.raises(FileNotFoundError):
        dryrun_mod._require_local_model_path(tmp_path / "does_not_exist")


# ── F. Renderer allowlist (project fields not leaked) ────────────────────────


def test_rendered_config_no_project_fields(tmp_path: Path):
    out = _render_sft_config(tmp_path)
    text = out.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert "experiment_class" not in data
    assert "train_dataset_path" not in data
    assert "eval_dataset_path" not in data
    assert "engineering_dryrun_only" not in data
    assert "sample_count" not in data
    assert "approval_required" not in data
    # Native LLaMA-Factory fields retained.
    assert "model_name_or_path" in data
    assert "dataset" in data
    assert "output_dir" in data


def test_rendered_dataset_conversion(tmp_path: Path):
    out = _render_sft_config(tmp_path)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["dataset"] == "phase04_dryrun_sft_1row"


def test_dpo_rendered_beta_config_no_project_fields(tmp_path: Path):
    src = TRAINING / "phase04_dpo_qwen3_1_7b_beta_0_1.yaml"
    out = tmp_path / "rendered_dpo.yaml"
    dryrun_mod._generate_dryrun_config(
        src, out, Path("experiments/phase04/dryrun/dpo_1_7b_beta_0_1"), "dpo",
        extra_overrides={"pref_beta": 0.1, "pref_loss": "sigmoid"},
    )
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert "experiment_class" not in data
    assert "train_dataset_path" not in data
    assert data["dataset"] == "phase04_dryrun_dpo_1pair"
    assert data["pref_beta"] == 0.1


# ── H. Artifact-based PASS ───────────────────────────────────────────────────


def test_adapter_artifacts_present(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    ok, missing = dryrun_mod._adapter_artifacts_present(out)
    assert not ok
    assert len(missing) == 2
    (out / "adapter_config.json").write_text("{}", encoding="utf-8")
    (out / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    ok, missing = dryrun_mod._adapter_artifacts_present(out)
    assert ok
    assert missing == []


# ── I. Eval matrix 12 runs ───────────────────────────────────────────────────


def test_matrix_12_runs_distribution():
    runs = build_phase04_eval_matrix()
    assert len(runs) == 12
    assert sum(1 for r in runs if r.stage == "sft") == 4
    assert sum(1 for r in runs if r.stage == "dpo" and r.pref_beta == 0.1) == 4
    assert sum(1 for r in runs if r.stage == "dpo" and r.pref_beta == 0.3) == 4
    # U/S each paired for every (model, stage, beta).
    for r in runs:
        assert r.variant in ("u", "s")


def test_manifest_retains_canonical_model_id(tmp_path: Path, monkeypatch):
    # Verify _update_sft_manifest records canonical_model_id + resolved path.
    monkeypatch.setattr(dryrun_mod, "MANIFEST_PATH", tmp_path / "manifest.json")
    dryrun_mod._update_sft_manifest(
        sft_passed=True, sft_sample_count=1, sft_max_steps=2, approval_file="a.json",
        sft_config="c.yaml", sft_output_dir=Path("o"), sft_train_dataset="t",
        sft_eval_dataset="e", canonical_model_id="Qwen/Qwen3-1.7B-Base",
        resolved_model_path="models/hf_cache/Qwen_Qwen3-1.7B-Base",
    )
    m = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert m["canonical_model_id"] == "Qwen/Qwen3-1.7B-Base"
    assert "0.6B" not in m["canonical_model_id"]
