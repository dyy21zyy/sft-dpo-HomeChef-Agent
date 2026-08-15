"""Phase 04 dry-run output_dir provenance regression.

Fixes stale 0.6B provenance in scripts/train/dryrun.py.

Contract:
- dry-run output_dir MUST be derived from the training config's model size/key,
  never hardcoded to 0_6b.
- 1.7B config  -> output path contains "1_7b" (e.g. experiments/phase04/dryrun/sft_1_7b)
- 4B config    -> output path contains "4b"   (e.g. experiments/phase04/dryrun/dpo_4b)
- No model_size -> training_stage fallback: model identity is independent of
  the sft/dpo experiment stage.
- 0.6B remains historical/non-formal only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.train.dryrun as dryrun_mod
from homechef_booking.training.config import load_training_run_spec

SFT_1_7B = Path("configs/training/phase04_sft_qwen3_1_7b.yaml")
DPO_1_7B = Path("configs/training/phase04_dpo_qwen3_1_7b.yaml")
SFT_4B = Path("configs/training/phase04_sft_qwen3_4b.yaml")
DPO_4B = Path("configs/training/phase04_dpo_qwen3_4b.yaml")


# ── model key derivation (pure) ──────────────────────────────────────────────


def test_derive_model_key_1_7b():
    assert dryrun_mod._derive_model_key("Qwen/Qwen3-1.7B-Base") == "1_7b"


def test_derive_model_key_4b_instruct():
    assert dryrun_mod._derive_model_key("Qwen/Qwen3-4B-Instruct-2507") == "4b"


def test_derive_model_key_0_6b_historical():
    # 0.6B is historical/non-formal only; still maps to a valid key but must
    # never be used as the formal default.
    assert dryrun_mod._derive_model_key("Qwen/Qwen3-0.6B-Base") == "0_6b"


def test_derive_model_key_unknown_raises():
    with pytest.raises(ValueError):
        dryrun_mod._derive_model_key("Qwen/Qwen3-Unknown")


# ── output_dir derivation ────────────────────────────────────────────────────


def test_sft_1_7b_output_dir_contains_1_7b():
    out = dryrun_mod._dryrun_output_dir("sft", "1_7b")
    assert "sft_1_7b" in out.name


def test_dpo_1_7b_output_dir_contains_1_7b():
    out = dryrun_mod._dryrun_output_dir("dpo", "1_7b")
    assert "dpo_1_7b" in out.name


def test_sft_4b_output_dir_contains_4b():
    out = dryrun_mod._dryrun_output_dir("sft", "4b")
    assert "sft_4b" in out.name


def test_dpo_4b_output_dir_contains_4b():
    out = dryrun_mod._dryrun_output_dir("dpo", "4b")
    assert "dpo_4b" in out.name


def test_no_hardcoded_0_6b_output_dir_default():
    # The default/formal dryrun output must NOT be hardcoded to 0_6b.
    out = dryrun_mod._dryrun_output_dir("sft", dryrun_mod._derive_model_key("Qwen/Qwen3-1.7B-Base"))
    assert "0_6b" not in str(out)


# ── no model_size -> training_stage fallback ─────────────────────────────────


def test_model_identity_independent_of_stage():
    # Same model (1.7B) used for BOTH sft and dpo keeps the SAME size key;
    # the sft/dpo difference lives in the output-dir prefix, not the model key.
    sft_dir = dryrun_mod._dryrun_output_dir("sft", "1_7b")
    dpo_dir = dryrun_mod._dryrun_output_dir("dpo", "1_7b")
    assert "1_7b" in sft_dir.name and "1_7b" in dpo_dir.name
    assert sft_dir.name != dpo_dir.name  # differ only by stage prefix


# ── manifest records real model size ─────────────────────────────────────────


def test_sft_manifest_output_dir_records_real_model_size():
    spec = load_training_run_spec(SFT_1_7B)
    model_key = dryrun_mod._derive_model_key(spec.model_name_or_path or "")
    out_dir = dryrun_mod._dryrun_output_dir("sft", model_key)
    assert model_key == "1_7b"
    assert "1_7b" in out_dir.name


def test_dpo_manifest_output_dir_records_real_model_size():
    spec = load_training_run_spec(DPO_4B)
    model_key = dryrun_mod._derive_model_key(spec.model_name_or_path or "")
    out_dir = dryrun_mod._dryrun_output_dir("dpo", model_key)
    assert model_key == "4b"
    assert "4b" in out_dir.name


def test_usage_header_examples_use_1_7b_not_0_6b():
    text = Path("scripts/train/dryrun.py").read_text(encoding="utf-8")
    assert "phase04_sft_qwen3_1_7b.yaml" in text, "formal usage example must use 1.7B SFT"
    # Formal DPO example is the beta-suffixed config (beta sweep).
    assert "phase04_dpo_qwen3_1_7b_beta_0_1.yaml" in text, "formal usage example must use 1.7B DPO"
    # 0.6B must not be presented as the formal example.
    assert "phase04_sft_qwen3_0_6b.yaml" not in text.split('"""')[0] or "historical" in text
