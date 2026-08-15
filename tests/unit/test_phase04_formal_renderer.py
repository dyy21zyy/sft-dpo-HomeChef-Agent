"""Phase 04 formal renderer tests (TDD RED first).

The current scripts/train/render_config.py dumps the full TrainingRunSpec
(including train_dataset_path/eval_dataset_path/experiment_class/etc.) and
MISSING the LLaMA-Factory dataset registry names (dataset/eval_dataset).

Contract:
- 1.7B SFT resolved: dataset=homechef_sft_v03_train, eval_dataset=homechef_sft_v03_val,
  no project-only metadata, do_train/do_eval true, model/template preserved.
- 4B SFT resolved: same dataset mapping, model=Qwen3-4B-Instruct-2507,
  template=qwen3_nothink.
- 4 DPO configs: dataset=homechef_dpo_v03_train, eval_dataset=homechef_dpo_v03_val,
  adapter present, pref_beta 0.1/0.3 correct, no project-only metadata.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from homechef_booking.training.config import load_training_run_spec
from homechef_booking.training.renderer import (
    Phase04TrainingConfigError,
    resolve_dataset_registry_name,
)
from scripts.train import render_config

_T = Path("configs/training")

SFT_1_7B = _T / "phase04_sft_qwen3_1_7b.yaml"
SFT_4B = _T / "phase04_sft_qwen3_4b.yaml"
DPO_1_7B_01 = _T / "phase04_dpo_qwen3_1_7b_beta_0_1.yaml"
DPO_1_7B_03 = _T / "phase04_dpo_qwen3_1_7b_beta_0_3.yaml"
DPO_4B_01 = _T / "phase04_dpo_qwen3_4b_beta_0_1.yaml"
DPO_4B_03 = _T / "phase04_dpo_qwen3_4b_beta_0_3.yaml"

_FORBIDDEN = ("train_dataset_path", "eval_dataset_path", "experiment_class",
              "engineering_dryrun_only", "approval_required", "sample_count")


def _render(path: Path) -> dict:
    """Render a formal config to a resolved dict via the renderer."""
    spec = load_training_run_spec(path)
    return render_config.render_training_spec(spec)


def _assert_no_project_metadata(resolved: dict):
    for field in _FORBIDDEN:
        assert field not in resolved, f"project-only field leaked: {field}"


# ── A. 1.7B SFT render ───────────────────────────────────────────────────────


def test_sft_1_7b_dataset_mapping():
    r = _render(SFT_1_7B)
    assert r["dataset"] == "homechef_sft_v03_train"
    assert r["eval_dataset"] == "homechef_sft_v03_val"


def test_sft_1_7b_no_project_metadata():
    r = _render(SFT_1_7B)
    _assert_no_project_metadata(r)


def test_sft_1_7b_execution_and_identity():
    r = _render(SFT_1_7B)
    assert r["do_train"] is True
    assert r["do_eval"] is True
    assert r["model_name_or_path"] == "Qwen/Qwen3-1.7B-Base"
    assert r["template"] == "qwen3"


# ── B. 4B SFT render ─────────────────────────────────────────────────────────


def test_sft_4b_dataset_mapping():
    r = _render(SFT_4B)
    assert r["dataset"] == "homechef_sft_v03_train"
    assert r["eval_dataset"] == "homechef_sft_v03_val"


def test_sft_4b_identity_and_template():
    r = _render(SFT_4B)
    assert r["model_name_or_path"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert r["template"] == "qwen3_nothink"
    assert r["do_train"] is True


# ── C. 4 DPO configs ─────────────────────────────────────────────────────────


def test_dpo_dataset_mapping_all_betas():
    for path in (DPO_1_7B_01, DPO_1_7B_03, DPO_4B_01, DPO_4B_03):
        r = _render(path)
        assert r["dataset"] == "homechef_dpo_v03_train", f"{path.name} dataset"
        assert r["eval_dataset"] == "homechef_dpo_v03_val", f"{path.name} eval_dataset"
        assert r.get("adapter_name_or_path"), f"{path.name} adapter missing"
        _assert_no_project_metadata(r)


def test_dpo_pref_beta_values():
    assert _render(DPO_1_7B_01)["pref_beta"] == 0.1
    assert _render(DPO_1_7B_03)["pref_beta"] == 0.3
    assert _render(DPO_4B_01)["pref_beta"] == 0.1
    assert _render(DPO_4B_03)["pref_beta"] == 0.3


# ── Resolver fail-closed ─────────────────────────────────────────────────────


def test_resolver_returns_registry_name():
    assert resolve_dataset_registry_name(
        "data/processed/sft/v0.3/train.jsonl") == "homechef_sft_v03_train"
    assert resolve_dataset_registry_name(
        "data/processed/dpo/v0.3/val.jsonl") == "homechef_dpo_v03_val"


def test_resolver_no_match_raises(tmp_path: Path):
    info = tmp_path / "dataset_info.json"
    info.write_text(yaml.dump({"a": {"file_name": "processed/sft/v0.3/train.jsonl"}}), encoding="utf-8")
    with pytest.raises(Phase04TrainingConfigError):
        resolve_dataset_registry_name("data/processed/other/x.jsonl", info)


def test_resolver_multiple_match_raises(tmp_path: Path):
    info = tmp_path / "dataset_info.json"
    info.write_text(yaml.dump({
        "a": {"file_name": "processed/sft/v0.3/train.jsonl"},
        "b": {"file_name": "processed/sft/v0.3/train.jsonl"},
    }), encoding="utf-8")
    with pytest.raises(Phase04TrainingConfigError):
        resolve_dataset_registry_name("data/processed/sft/v0.3/train.jsonl", info)


def test_resolver_strips_data_prefix():
    # project path has "data/" prefix, registry file_name does not.
    assert resolve_dataset_registry_name(
        "data/processed/sft/v0.3/val.jsonl") == "homechef_sft_v03_val"


# ── Fail-closed gate ─────────────────────────────────────────────────────────


def test_gate_passes_on_valid_resolved():
    r = _render(SFT_1_7B)
    assert render_config.assert_rendered_config(r, stage="sft") is True


def test_gate_rejects_missing_do_train():
    r = _render(SFT_1_7B)
    r["do_train"] = False
    with pytest.raises(Phase04TrainingConfigError):
        render_config.assert_rendered_config(r, stage="sft")


def test_gate_rejects_leaked_project_field():
    r = _render(SFT_1_7B)
    r["experiment_class"] = "formal"
    with pytest.raises(Phase04TrainingConfigError):
        render_config.assert_rendered_config(r, stage="sft")


# ── 6-config render gate ─────────────────────────────────────────────────────


def test_six_formal_configs_render():
    for path in (SFT_1_7B, SFT_4B, DPO_1_7B_01, DPO_1_7B_03, DPO_4B_01, DPO_4B_03):
        r = _render(path)
        stage = "dpo" if "dpo" in path.name else "sft"
        assert render_config.assert_rendered_config(r, stage=stage) is True


# ── LLaMA-Factory parser smoke (no model load) ───────────────────────────────


@pytest.mark.skipif(
    not Path("experiments/phase04/dryrun/phase04_dryrun_sft_1row.yaml").exists(),
    reason="Requires a rendered dry-run config to parse; run dry-run first",
)
def test_llamafactory_parse_rendered_sft(tmp_path: Path):
    """Parse a rendered formal-like SFT config through LLaMA-Factory's real
    parser (no model load / no training / no GPU). Confirms no unsupported args.
    """
    llamafactory = pytest.importorskip("llamafactory.hparams")
    # Render the formal SFT config to a temp YAML and parse it.
    r = _render(SFT_1_7B)
    yaml_path = tmp_path / "resolved.yaml"
    yaml_path.write_text(yaml.dump(r, sort_keys=False), encoding="utf-8")
    from llamafactory.hparams import get_train_args

    model_args, data_args, training_args, finetuning_args, gen_args = get_train_args(
        ["train", str(yaml_path)]
    )
    assert training_args.do_train is True
    assert data_args.stage == "sft"
    assert data_args.dataset == "homechef_sft_v03_train"
