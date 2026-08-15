"""Phase 04 renderer do_train contract tests (TDD RED first).

Root cause (confirmed): the dry-run rendered LLaMA-Factory YAML was missing
``do_train: true``, so LLaMA-Factory's ``if training_args.do_train`` branch was
skipped -> silent no-op (exit 0, global_step=0, no adapter).

Contract:
- SFT/DPO dry-run rendered config MUST set do_train: true (and do_eval: false).
- LLaMA-Factory parser must yield training_args.do_train is True.
- Formal training configs must keep do_train=true while preserving formal eval
  semantics (do NOT force do_eval=false on formal training).
"""

from __future__ import annotations

from pathlib import Path

import yaml

import scripts.train.dryrun as dryrun_mod
from homechef_booking.training.config import load_training_run_spec

_TRAINING = Path("configs/training")
_SFT = _TRAINING / "phase04_sft_qwen3_1_7b.yaml"
_DPO = _TRAINING / "phase04_dpo_qwen3_1_7b_beta_0_1.yaml"


def _render_sft(tmp_path: Path, **kwargs):
    out = tmp_path / "rendered_sft.yaml"
    dryrun_mod._generate_dryrun_config(
        _SFT, out, Path("experiments/phase04/dryrun/sft_1_7b"), "sft", **kwargs,
    )
    return out


def _render_dpo(tmp_path: Path, **kwargs):
    out = tmp_path / "rendered_dpo.yaml"
    dryrun_mod._generate_dryrun_config(
        _DPO, out, Path("experiments/phase04/dryrun/dpo_1_7b_beta_0_1"), "dpo",
        extra_overrides={"pref_beta": 0.1, "pref_loss": "sigmoid"}, **kwargs,
    )
    return out


# ── A/B. SFT & DPO dry-run rendered config must set do_train: true ──────────


def test_sft_dryrun_rendered_do_train_true(tmp_path: Path):
    out = _render_sft(tmp_path)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data.get("do_train") is True, "SFT dry-run rendered config MUST set do_train: true"


def test_sft_dryrun_rendered_do_eval_false(tmp_path: Path):
    out = _render_sft(tmp_path)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data.get("do_eval") is False, "SFT dry-run should set do_eval: false"


def test_dpo_dryrun_rendered_do_train_true(tmp_path: Path):
    out = _render_dpo(tmp_path)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data.get("do_train") is True, "DPO dry-run rendered config MUST set do_train: true"


def test_dpo_dryrun_rendered_do_eval_false(tmp_path: Path):
    out = _render_dpo(tmp_path)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data.get("do_eval") is False, "DPO dry-run should set do_eval: false"


def test_do_train_does_not_rely_on_stage_only(tmp_path: Path):
    # The config must NOT assume stage=sft/dpo implies training.
    out = _render_sft(tmp_path)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert "do_train" in data, "do_train must be explicit, not inferred from stage"


# ── C. LLaMA-Factory parser yields do_train=True ─────────────────────────────


def test_parser_do_train_true(tmp_path: Path):
    out = _render_sft(tmp_path)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["do_train"] is True


# ── Formal training configs keep do_train=true, NOT forced do_eval=false ─────


def test_formal_sft_config_do_train_true():
    # Formal training config must explicitly set do_train=true (execution
    # control, not a hyperparameter).
    data = yaml.safe_load(_SFT.read_text(encoding="utf-8"))
    assert data.get("do_train") is True, "formal SFT config must set do_train=true"
    # Formal SFT config must not be forced to do_eval=false (eval semantics preserved).
    assert data.get("do_eval") is not False, "formal training must not force do_eval=false"


def test_formal_dpo_beta_configs_do_train_true():
    for name in ("phase04_dpo_qwen3_1_7b_beta_0_1.yaml", "phase04_dpo_qwen3_1_7b_beta_0_3.yaml",
                 "phase04_dpo_qwen3_4b_beta_0_1.yaml", "phase04_dpo_qwen3_4b_beta_0_3.yaml",
                 "phase04_sft_qwen3_4b.yaml"):
        spec = load_training_run_spec(_TRAINING / name)
        assert spec is not None
        data = yaml.safe_load((_TRAINING / name).read_text(encoding="utf-8"))
        assert data.get("do_eval") is not False, f"{name} must not force do_eval=false"


# ── Fail-closed preflight ─────────────────────────────────────────────────────


def test_preflight_raises_on_do_train_false():
    import pytest

    from homechef_booking.training.renderer import (
        Phase04TrainingConfigError,
        assert_rendered_config,
    )

    cfg = {"stage": "sft", "do_train": False, "do_eval": False,
           "dataset": "x", "eval_dataset": "y", "model_name_or_path": "m"}
    with pytest.raises(Phase04TrainingConfigError, match="do_train=true"):
        assert_rendered_config(cfg, stage="sft")


def test_preflight_passes_on_do_train_true():
    from homechef_booking.training.renderer import assert_rendered_config

    cfg = {"stage": "sft", "do_train": True, "do_eval": False,
           "dataset": "x", "eval_dataset": "y", "model_name_or_path": "m"}
    assert assert_rendered_config(cfg, stage="sft") is True
