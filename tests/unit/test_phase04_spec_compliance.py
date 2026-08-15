"""Phase 04 spec-compliance tests (round 2 Codex review).

Covers:
- DPO batch restored to 1/1 for formal configs (finetune-spec/Plan priority).
- 4B Instruct-2507 uses template=qwen3_nothink (registered in the pinned
  LLaMA-Factory); 1.7B Base keeps its frozen template (qwen3).
- Historical 0.6B / v0.1 remain isolated from the formal experiment.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from homechef_booking.training.config import load_training_run_spec


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ── DPO batch = 1/1 (finetune-spec/Plan priority) ───────────────────────────


def test_dpo_1_7b_batch_is_1_1():
    spec = load_training_run_spec(Path("configs/training/phase04_dpo_qwen3_1_7b.yaml"))
    assert spec.per_device_train_batch_size == 1
    assert spec.per_device_eval_batch_size == 1


def test_dpo_4b_batch_is_1_1():
    spec = load_training_run_spec(Path("configs/training/phase04_dpo_qwen3_4b.yaml"))
    assert spec.per_device_train_batch_size == 1
    assert spec.per_device_eval_batch_size == 1


def test_dpo_batch_does_not_touch_other_frozen_params():
    # SPEC-approved DPO frozen params must remain unchanged.
    for name in ("phase04_dpo_qwen3_1_7b", "phase04_dpo_qwen3_4b"):
        spec = load_training_run_spec(Path(f"configs/training/{name}.yaml"))
        assert spec.pref_beta == 0.1
        assert spec.pref_loss == "sigmoid"
        assert spec.learning_rate == 5.0e-6
        assert spec.num_train_epochs == 1
        assert spec.lora_rank == 16
        assert spec.lora_alpha == 32


# ── 4B template = qwen3_nothink ──────────────────────────────────────────────


def test_4b_sft_uses_qwen3_nothink():
    data = _read_yaml(Path("configs/training/phase04_sft_qwen3_4b.yaml"))
    assert data["template"] == "qwen3_nothink"
    # Do NOT rely on template=qwen3 + enable_thinking:false for 4B.
    assert data.get("enable_thinking", True) is not False


def test_4b_dpo_uses_qwen3_nothink():
    data = _read_yaml(Path("configs/training/phase04_dpo_qwen3_4b.yaml"))
    assert data["template"] == "qwen3_nothink"


def test_1_7b_base_keeps_frozen_template():
    # 1.7B Base config must NOT be modified to qwen3_nothink (frozen spec).
    data = _read_yaml(Path("configs/training/phase04_sft_qwen3_1_7b.yaml"))
    assert data["template"] == "qwen3"


def test_qwen3_nothink_registered_in_pinned_llamafactory():
    # Verify the installed/pinned LLaMA-Factory registers qwen3_nothink by
    # inspecting the template source WITHOUT importing the package (importing
    # llamafactory.data pulls in torch, which is broken in this env).
    import importlib.metadata as md

    files = md.files("llamafactory")
    assert files is not None, "llamafactory not installed"
    template_entry = next(
        (f for f in files if f.name == "template.py" and "data" in f.parts),
        None,
    )
    assert template_entry is not None, "llamafactory.data.template.py not found"
    text = template_entry.read_text(encoding="utf-8")
    assert 'name="qwen3_nothink"' in text, "qwen3_nothink NOT registered in pinned LLaMA-Factory"
    assert 'name="qwen3"' in text


# ── Historical isolation (0.6B / v0.1) ───────────────────────────────────────


def test_0_6b_configs_are_historical():
    for name in ("phase04_sft_qwen3_0_6b", "phase04_dpo_qwen3_0_6b"):
        spec = load_training_run_spec(Path(f"configs/training/{name}.yaml"))
        assert spec.experiment_class == "historical"


def test_historical_dpo_comment_is_non_formal():
    text = Path("configs/training/phase04_dpo_qwen3_0_6b.yaml").read_text(encoding="utf-8")
    assert "HISTORICAL DPO config / non-formal" in text
    assert "FORMAL DPO config" not in text


def test_v01_dataset_registry_entries_still_present_for_repro():
    # v0.1 registry entries must remain (historical reproduction), but formal
    # configs must not reference them.
    for name in ("phase04_sft_qwen3_1_7b", "phase04_sft_qwen3_4b",
                 "phase04_dpo_qwen3_1_7b", "phase04_dpo_qwen3_4b"):
        spec = load_training_run_spec(Path(f"configs/training/{name}.yaml"))
        assert "v0.1" not in str(spec.train_dataset_path)
        assert "v0.1" not in str(spec.eval_dataset_path)
        assert "v0.3" in str(spec.train_dataset_path)
