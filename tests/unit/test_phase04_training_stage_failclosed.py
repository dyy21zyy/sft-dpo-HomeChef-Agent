"""Phase 04 training_stage fail-closed regression (round 4).

Contract (Codex round-4 review):
- training_stage is an EXPERIMENT-STAGE property. It MUST be explicitly
  provided by the formal Phase04 backend config (sft | dpo).
- Phase04HFBackend MUST NOT derive training_stage from model_key / model_size /
  base_model_id / adapter-path name string, and MUST NOT default to "sft".
- Missing or invalid training_stage -> HARD FAIL / fail-closed. Legal values
  are ONLY: "sft", "dpo".

TDD: these tests are written BEFORE the production fix. Cases A and D should be
RED against the current backend (which still has the model-key fallback).
"""

from __future__ import annotations

import pytest

from homechef_booking.inference.phase04_hf_backend import (
    _MODEL_KEY_TO_META,
    Phase04HFConfig,
    Phase04HFTransformersBackend,
)

_VALID_ADAPTER = "experiments/phase04/dpo_qwen3_1_7b/checkpoint-best"


def _cfg(**overrides) -> Phase04HFConfig:
    base = {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "adapter_name_or_path": _VALID_ADAPTER,
        "use_structured_output": False,
        "device": "cpu",
        "model_key": "1_7b",
        "model_size": "1.7B",
    }
    base.update(overrides)
    return Phase04HFConfig(**base)


# ── Case A: missing training_stage must FAIL CLOSED ──────────────────────────


def test_missing_training_stage_fails_closed():
    cfg = _cfg(training_stage=None)
    errors = cfg.validate_for_run()
    assert any("training_stage" in e for e in errors), (
        "missing training_stage must be a validation error (currently silently defaults to sft)"
    )


# ── Case D: invalid training_stage must FAIL CLOSED ──────────────────────────


@pytest.mark.parametrize("bad", ["unknown", "SFT", "dpo ", "  ", "pretrain"])
def test_invalid_training_stage_fails_closed(bad):
    cfg = _cfg(training_stage=bad)
    errors = cfg.validate_for_run()
    assert any("training_stage" in e for e in errors), (
        f"invalid training_stage={bad!r} must be a validation error"
    )


def test_valid_training_stages_are_accepted():
    for good in ("sft", "dpo"):
        cfg = _cfg(training_stage=good)
        assert cfg.validate_for_run() == []


# ── Case B: model_key=1_7b + training_stage=dpo -> dpo (not sft) ─────────────


def test_1_7b_dpo_not_overridden_by_model_key():
    cfg = _cfg(model_key="1_7b", training_stage="dpo")
    backend = Phase04HFTransformersBackend()
    backend._config = cfg
    backend._config.loaded_adapter = _VALID_ADAPTER
    prov = backend._provenance_kwargs()
    assert prov["training_stage"] == "dpo"
    assert prov["model_size"] == "1.7B"


# ── Case C: model_key=4b + training_stage=dpo -> dpo ─────────────────────────


def test_4b_dpo_not_overridden_by_model_key():
    cfg = _cfg(model_id="Qwen/Qwen3-4B-Instruct-2507", model_key="4b", model_size="4B",
               training_stage="dpo")
    backend = Phase04HFTransformersBackend()
    backend._config = cfg
    backend._config.loaded_adapter = "experiments/phase04/dpo_qwen3_4b/checkpoint-best"
    prov = backend._provenance_kwargs()
    assert prov["training_stage"] == "dpo"
    assert prov["model_size"] == "4B"


def test_sft_stage_is_sft():
    cfg = _cfg(model_key="1_7b", training_stage="sft")
    backend = Phase04HFTransformersBackend()
    backend._config = cfg
    backend._config.loaded_adapter = "experiments/phase04/sft_qwen3_1_7b/checkpoint-best"
    prov = backend._provenance_kwargs()
    assert prov["training_stage"] == "sft"


# ── model-key mapping must NOT contain training_stage ────────────────────────


def test_model_key_meta_has_no_training_stage():
    # Per round-4 contract: the model-key metadata must not carry training_stage.
    for meta in _MODEL_KEY_TO_META.values():
        assert "training_stage" not in meta, (
            "model-key metadata must not contain training_stage (stage is not model identity)"
        )
