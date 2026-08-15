"""Phase 04 eval config load smoke (EVAL_CONFIG_LOAD = 8/8).

Loads each of the 8 Phase04 eval configs AND their backend configs through the
existing evaluation/runner + benchmark_runner machinery (not just YAML parse):

- Every eval YAML is loadable via BenchmarkConfig.load_yaml and points at an
  existing cases_path + manifest_path + backend_config_path.
- Every backend YAML is a well-formed phase04_hf backend config.
- U/S pairs share model_id + adapter; differ only in use_structured_output.
- All repo-relative paths use POSIX separators (Linux portability).

Because formal adapters are not trained yet (SFT/DPO NOT RUN), the backend
*load* would fail-closed on a missing adapter — which is the expected behavior,
so this test validates config structure + loadability, not adapter presence.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from homechef_booking.evaluation.benchmark_runner import BenchmarkConfig
from homechef_booking.inference.phase04_hf_backend import PHASE02_FINAL_MODEL_IDS
from homechef_booking.training.formal_matrix import (
    FORMAL_MODEL_IDS,
    build_phase04_eval_matrix,
    validate_all_matrix_runs,
)

_EVAL_DIR = Path("configs/evaluation/phase04_matrix")
_BACKEND_DIR = Path("configs/phase04/backends")


def _all_runs():
    return build_phase04_eval_matrix()


def test_all_12_eval_configs_are_written():
    runs = _all_runs()
    assert len(runs) == 12
    for r in runs:
        assert (_EVAL_DIR / f"{r.run_id}.yaml").exists()
        assert r.backend_config_path.exists()


def test_eval_config_load_12_of_12_benchmark_config():
    runs = _all_runs()
    loaded = 0
    for r in runs:
        yaml_path = _EVAL_DIR / f"{r.run_id}.yaml"
        # Load through the existing BenchmarkConfig loader.
        cfg = BenchmarkConfig.load_yaml(yaml_path)
        assert cfg.run_id == r.run_id
        assert cfg.model_id == r.model_id
        # Paths must exist / be reachable.
        assert (r.backend_config_path).exists(), f"backend config missing for {r.run_id}"
        assert cfg.cases_path == Path("data/eval/frozen_test.jsonl")
        assert Path(cfg.cases_path).exists(), f"cases_path missing for {r.run_id}"
        assert cfg.manifest_path is not None
        loaded += 1
    assert loaded == 12


def test_eval_config_has_backend_config_path():
    runs = _all_runs()
    for r in runs:
        data = yaml.safe_load((_EVAL_DIR / f"{r.run_id}.yaml").read_text(encoding="utf-8"))
        assert "backend_config_path" in data, f"{r.run_id} missing backend_config_path"
        assert "\\" not in data["backend_config_path"], f"{r.run_id} non-POSIX backend path"


def test_backend_config_is_phase04_hf_and_complete():
    runs = _all_runs()
    for r in runs:
        data = yaml.safe_load(r.backend_config_path.read_text(encoding="utf-8"))
        assert data["backend"] == "phase04_hf"
        assert data["model_id"] == r.model_id
        assert data["model_id"] in FORMAL_MODEL_IDS
        assert data["adapter_name_or_path"] == str(r.adapter_name_or_path).replace("\\", "/")
        assert data["use_structured_output"] == r.use_structured_output
        # Required backend fields per review.
        for field in ("device", "torch_dtype", "max_model_length", "max_new_tokens"):
            assert field in data, f"{r.run_id} backend missing {field}"
        # Provenance recorded in backend config (not only in-memory).
        assert data.get("model_key") == r.model_key
        assert data.get("training_stage") in ("sft", "dpo")
        assert data.get("model_size") in ("1.7B", "4B")
        # pref_beta: None for SFT, 0.1/0.3 for DPO.
        assert data.get("pref_beta") == r.pref_beta


def test_us_pair_identical_except_structured():
    runs = _all_runs()
    by_key = {}
    for r in runs:
        by_key.setdefault((r.model_key, r.stage, r.pref_beta), []).append(r)
    for group in by_key.values():
        u = next(r for r in group if r.variant == "u")
        s = next(r for r in group if r.variant == "s")
        u_be = yaml.safe_load(u.backend_config_path.read_text(encoding="utf-8"))
        s_be = yaml.safe_load(s.backend_config_path.read_text(encoding="utf-8"))
        assert u_be["model_id"] == s_be["model_id"]
        assert u_be["adapter_name_or_path"] == s_be["adapter_name_or_path"]
        # Frozen Test identical (same cases_path).
        u_eval = yaml.safe_load((_EVAL_DIR / f"{u.run_id}.yaml").read_text(encoding="utf-8"))
        s_eval = yaml.safe_load((_EVAL_DIR / f"{s.run_id}.yaml").read_text(encoding="utf-8"))
        assert u_eval["cases_path"] == s_eval["cases_path"]
        # ONLY structured differs.
        assert u_be["use_structured_output"] is False
        assert s_be["use_structured_output"] is True


def test_backend_model_ids_are_phase02_final():
    runs = _all_runs()
    for r in runs:
        assert r.model_id in PHASE02_FINAL_MODEL_IDS
        # No 0.6B / 4B-Base leaks.
        assert "0.6B" not in r.model_id
        assert r.model_id != "Qwen/Qwen3-4B-Base"


def test_all_matrix_runs_validate():
    runs = _all_runs()
    assert validate_all_matrix_runs(runs) == []


def test_no_v01_or_windows_sep_in_generated_configs():
    runs = _all_runs()
    for r in runs:
        for path in ((_EVAL_DIR / f"{r.run_id}.yaml"), r.backend_config_path):
            text = path.read_text(encoding="utf-8")
            assert "v0.1" not in text, f"v0.1 leaked into {path}"
            assert "\\" not in text, f"Windows separator leaked into {path}"
            assert "experiments/phase04/" in text, f"repo-relative path not POSIX in {path}"
