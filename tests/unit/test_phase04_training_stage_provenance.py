"""Phase 04 training_stage provenance regression (round 3).

Fixes: provenance training_stage must be an EXPERIMENT-STAGE property derived
from run.stage, NOT derived from model_key. Previously all DPO backend configs
wrongly reported training_stage: sft.

Assertions:
- All *dpo*_backend.yaml -> training_stage == "dpo".
- All *sft*_backend.yaml -> training_stage == "sft".
- Full 8-run: SFT stage count == 4, DPO stage count == 4.
- U/S pair training_stage identical (e.g. 1.7B DPO-U and DPO-S both "dpo").
- Provenance chain consistency:
    backend config.training_stage == GenerationResult.training_stage
      == case result.training_stage
- DPO provenance persistence: mock DPO backend -> generate -> serialize
  (mirroring runner.py) -> disk/case artifact has training_stage == "dpo".
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from homechef_booking.inference.backend import GenerationParams
from homechef_booking.inference.phase04_hf_backend import Phase04HFConfig
from homechef_booking.inference.response import GenerationResult
from homechef_booking.training.formal_matrix import (
    build_phase04_eval_matrix,
    validate_all_matrix_runs,
)

_EVAL_DIR = Path("configs/evaluation/phase04_matrix")
_BACKEND_DIR = Path("configs/phase04/backends")


@pytest.fixture(autouse=True)
def _fake_torch(monkeypatch):
    import contextlib

    fake = types.ModuleType("torch")
    fake.no_grad = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", fake)
    yield


def _backend_stages() -> dict[str, dict]:
    runs = build_phase04_eval_matrix()
    out = {}
    for r in runs:
        out[r.run_id] = yaml.safe_load(r.backend_config_path.read_text(encoding="utf-8"))
    return out


# ── DPO / SFT backend config stage mapping ───────────────────────────────────


def test_dpo_backend_configs_report_dpo():
    stages = _backend_stages()
    dpo_ids = [rid for rid in stages if "_dpo_" in rid]
    assert len(dpo_ids) == 8
    for rid in dpo_ids:
        assert stages[rid]["training_stage"] == "dpo", f"{rid} must be dpo, got {stages[rid]['training_stage']}"


def test_sft_backend_configs_report_sft():
    stages = _backend_stages()
    sft_ids = [rid for rid in stages if "_sft_" in rid]
    assert len(sft_ids) == 4
    for rid in sft_ids:
        assert stages[rid]["training_stage"] == "sft", f"{rid} must be sft, got {stages[rid]['training_stage']}"


def test_sft_dpo_backend_stage_counts_are_4_and_8():
    stages = _backend_stages()
    sft_count = sum(1 for b in stages.values() if b["training_stage"] == "sft")
    dpo_count = sum(1 for b in stages.values() if b["training_stage"] == "dpo")
    assert sft_count == 4
    assert dpo_count == 8
    assert sft_count + dpo_count == 12


def test_us_pair_training_stage_identical():
    runs = build_phase04_eval_matrix()
    by_key = {}
    for r in runs:
        by_key.setdefault((r.model_key, r.stage, r.pref_beta), []).append(r)
    for group in by_key.values():
        u = next(r for r in group if r.variant == "u")
        s = next(r for r in group if r.variant == "s")
        assert u.training_stage == s.training_stage
        assert u.training_stage == u.stage
        assert s.training_stage == s.stage


def test_specific_dpo_pairs_are_dpo():
    # DPO runs (beta 0.1/0.3) across both models.
    for rid in ("phase04_1_7b_dpo_beta_0_1_u", "phase04_1_7b_dpo_beta_0_1_s",
                "phase04_1_7b_dpo_beta_0_3_u", "phase04_1_7b_dpo_beta_0_3_s",
                "phase04_4b_dpo_beta_0_1_u", "phase04_4b_dpo_beta_0_1_s",
                "phase04_4b_dpo_beta_0_3_u", "phase04_4b_dpo_beta_0_3_s"):
        data = (_BACKEND_DIR / f"{rid}_backend.yaml").read_text(encoding="utf-8")
        assert "training_stage: dpo" in data, f"{rid} must be dpo"


def test_specific_sft_pairs_are_sft():
    for rid in ("phase04_1_7b_sft_u", "phase04_1_7b_sft_s",
                "phase04_4b_sft_u", "phase04_4b_sft_s"):
        data = (_BACKEND_DIR / f"{rid}_backend.yaml").read_text(encoding="utf-8")
        assert "training_stage: sft" in data, f"{rid} must be sft"


def test_training_stage_is_run_stage_not_model_key():
    # Direct contract: training_stage derives from run.stage, not model_key.
    runs = build_phase04_eval_matrix()
    for r in runs:
        assert r.training_stage == r.stage
        assert r.training_stage in ("sft", "dpo")
        # model_size must stay independent of training stage.
        if "1_7b" in r.model_key:
            assert r.model_size == "1.7B"
        else:
            assert r.model_size == "4B"


def test_no_dead_model_key_training_stage_mapping():
    # The removed dead mapping must not exist.
    import homechef_booking.training.formal_matrix as fm
    assert not hasattr(fm, "_MODEL_KEY_TRAINING_STAGE")


# ── Provenance chain: backend config -> GenerationResult -> case artifact ────


class _DPOBackend:
    """Minimal fake that mimics a phase04_hf backend loaded from a DPO config."""

    def __init__(self, cfg: Phase04HFConfig):
        self._config = cfg
        self._constraint_fn = None
        self._model = None
        self._tokenizer = None

    def _provenance_kwargs(self) -> dict:
        return {
            "base_model_id": self._config.model_id,
            "adapter_name_or_path": self._config.adapter_name_or_path,
            "training_stage": self._config.training_stage,
            "model_size": self._config.model_size,
            "use_structured_output": self._config.use_structured_output,
        }

    def generate(self, messages, params, case_id=None):
        return GenerationResult(
            case_id=case_id or "c1",
            backend_name="phase04_hf",
            raw_text=('{"action": "final", "booking_state": {}, '
                      '"chef_query_status": "not_checked", "info_complete": true, '
                      '"unrelated": false, "reply_type": "acknowledge_result"}'),
            finish_reason="stop",
            latency_ms=1.0,
            **self._provenance_kwargs(),
        )


def _serialize_like_runner(gen: GenerationResult) -> dict:
    """Mirror the enriched case-result serialization in evaluation/runner.py."""
    return {
        "raw_text": gen.raw_text,
        "finish_reason": gen.finish_reason,
        "error_type": gen.error_type,
        "error_message": gen.error_message,
        "latency_ms": gen.latency_ms,
        "base_model_id": gen.base_model_id,
        "adapter_name_or_path": gen.adapter_name_or_path,
        "training_stage": gen.training_stage,
        "model_size": gen.model_size,
        "use_structured_output": gen.use_structured_output,
    }


def test_generation_result_stage_matches_backend_config():
    # backend config.training_stage == GenerationResult.training_stage.
    cfg = Phase04HFConfig(
        model_id="Qwen/Qwen3-1.7B-Base",
        adapter_name_or_path="experiments/phase04/dpo_qwen3_1_7b/checkpoint-best",
        use_structured_output=True,
        device="cpu",
        model_key="1_7b",
        training_stage="dpo",
        model_size="1.7B",
        loaded_adapter="experiments/phase04/dpo_qwen3_1_7b/checkpoint-best",
    )
    backend = _DPOBackend(cfg)
    gen = backend.generate([{"role": "user", "content": "x"}], GenerationParams(), case_id="c1")
    assert gen.training_stage == "dpo"
    assert gen.training_stage == cfg.training_stage
    assert gen.model_size == "1.7B"


def test_dpo_provenance_persists_to_case_artifact():
    # DPO backend -> generate -> runner serialize -> case artifact
    #   training_stage == "dpo" (never "sft").
    cfg = Phase04HFConfig(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        adapter_name_or_path="experiments/phase04/dpo_qwen3_4b/checkpoint-best",
        use_structured_output=False,
        device="cpu",
        model_key="4b",
        training_stage="dpo",
        model_size="4B",
        loaded_adapter="experiments/phase04/dpo_qwen3_4b/checkpoint-best",
    )
    backend = _DPOBackend(cfg)
    gen = backend.generate([{"role": "user", "content": "x"}], GenerationParams(), case_id="c1")
    artifact = _serialize_like_runner(gen)
    assert artifact["training_stage"] == "dpo"
    assert artifact["training_stage"] == cfg.training_stage
    # No config=dpo / artifact=sft mismatch.
    assert not (artifact["training_stage"] == "sft")


def test_case_result_json_on_disk_has_dpo_stage(tmp_path):
    # Full: generate -> serialize -> write to disk -> read back == "dpo".
    cfg = Phase04HFConfig(
        model_id="Qwen/Qwen3-1.7B-Base",
        adapter_name_or_path="experiments/phase04/dpo_qwen3_1_7b/checkpoint-best",
        use_structured_output=True,
        device="cpu",
        model_key="1_7b",
        training_stage="dpo",
        model_size="1.7B",
        loaded_adapter="experiments/phase04/dpo_qwen3_1_7b/checkpoint-best",
    )
    backend = _DPOBackend(cfg)
    gen = backend.generate([{"role": "user", "content": "x"}], GenerationParams(), case_id="c1")
    entry = {"id": "c1", "generation_result": _serialize_like_runner(gen)}
    out = tmp_path / "case_results.json"
    out.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["generation_result"]["training_stage"] == "dpo"


def test_matrix_still_valid():
    runs = build_phase04_eval_matrix()
    assert validate_all_matrix_runs(runs) == []


def test_eval_configs_load_12_of_12():
    from homechef_booking.evaluation.benchmark_runner import BenchmarkConfig

    runs = build_phase04_eval_matrix()
    assert len(runs) == 12
    for r in runs:
        yaml_path = _EVAL_DIR / f"{r.run_id}.yaml"
        cfg = BenchmarkConfig.load_yaml(yaml_path)
        assert cfg.model_id == r.model_id
        assert r.backend_config_path.exists()
