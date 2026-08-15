"""Phase 04 Frozen Test evaluation matrix + comparison tests (TDD).

Covers:
- 12-cell exploratory matrix (run_ids, 6 adapters, U/S pairing, beta 0.1/0.3).
- Frozen Test count=120 + SHA gate.
- Missing adapter hard fail.
- --skip-complete logic.
- Summarizer: 16-cell comparison, deltas, case-level diff, exploratory selection.
  (Comparison/delta/selection are exercised with synthetic scorecards since real
  Phase04 scorecards require trained adapters; EVALUATION is NOT RUN here.)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from homechef_booking.inference.response import GenerationResult
from homechef_booking.training.formal_matrix import (
    FROZEN_TEST_PATH,
    FROZEN_TEST_SHA256,
    FROZEN_TEST_TOTAL_CASES,
    build_phase04_exploratory_matrix,
    validate_exploratory_matrix,
)
from scripts.eval import run_phase04_frozen_matrix as runner
from scripts.eval import summarize_phase04_matrix as summer

EXPECTED_IDS = {
    "phase04_1_7b_sft_u", "phase04_1_7b_sft_s",
    "phase04_1_7b_dpo_b01_u", "phase04_1_7b_dpo_b01_s",
    "phase04_1_7b_dpo_b03_u", "phase04_1_7b_dpo_b03_s",
    "phase04_4b_sft_u", "phase04_4b_sft_s",
    "phase04_4b_dpo_b01_u", "phase04_4b_dpo_b01_s",
    "phase04_4b_dpo_b03_u", "phase04_4b_dpo_b03_s",
}


def _matrix():
    return build_phase04_exploratory_matrix()


# ── Matrix structure ─────────────────────────────────────────────────────────


def test_matrix_count_12():
    assert len(_matrix()) == 12


def test_matrix_unique_run_ids_12():
    runs = _matrix()
    assert len({r.run_id for r in runs}) == 12
    assert {r.run_id for r in runs} == EXPECTED_IDS


def test_matrix_6_unique_adapters_each_us():
    runs = _matrix()
    adapters = {}
    for r in runs:
        adapters.setdefault(str(r.adapter_name_or_path), []).append(r)
    assert len(adapters) == 6
    for _adapter, group in adapters.items():
        assert {r.variant for r in group} == {"u", "s"}
        # U/S share the same adapter.
        assert len({r.adapter_name_or_path for r in group}) == 1


def test_matrix_dpo_betas_both_present():
    runs = _matrix()
    betas = {r.pref_beta for r in runs if r.stage == "dpo"}
    assert betas == {0.1, 0.3}


def test_matrix_1_7b_and_4b_each_6_cells():
    runs = _matrix()
    assert sum(1 for r in runs if r.model_key == "1_7b") == 6
    assert sum(1 for r in runs if r.model_key == "4b") == 6


def test_matrix_us_pair_same_adapter_only_structured_differs():
    runs = _matrix()
    by_adapter = {}
    for r in runs:
        by_adapter.setdefault(str(r.adapter_name_or_path), []).append(r)
    for group in by_adapter.values():
        u = next(r for r in group if r.variant == "u")
        s = next(r for r in group if r.variant == "s")
        assert u.adapter_name_or_path == s.adapter_name_or_path
        assert u.model_id == s.model_id
        assert u.use_structured_output is False
        assert s.use_structured_output is True
        # Only intentional runtime difference is the structured flag.
        assert u.stage == s.stage and u.pref_beta == s.pref_beta


def test_matrix_validation_passes():
    assert validate_exploratory_matrix(_matrix()) == []


# ── Frozen Test ──────────────────────────────────────────────────────────────


def test_frozen_test_count_120():
    n = sum(1 for line in FROZEN_TEST_PATH.open(encoding="utf-8") if line.strip())
    assert n == FROZEN_TEST_TOTAL_CASES == 120


def test_frozen_test_sha_match():
    sha = hashlib.sha256(FROZEN_TEST_PATH.read_bytes()).hexdigest()
    assert sha == FROZEN_TEST_SHA256


# ── Runner fail-closed + skip-complete ───────────────────────────────────────


def test_missing_adapter_hard_fail(tmp_path: Path):
    # No adapter dir exists -> _run_cell must raise (no Base fallback).
    runs = _matrix()
    run = runs[0]
    assert not run.adapter_name_or_path.exists() or not (run.adapter_name_or_path / "adapter_config.json").exists()
    import pytest

    with pytest.raises(FileNotFoundError, match="adapter"):
        runner._run_cell(run, [], "sha")


def test_is_complete_detects_incomplete(tmp_path: Path):
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    assert runner._is_complete(run_dir) is False


def _write_cell(run_dir: Path, total=120, success=120, failure=0, raw_non_null=120):
    run_dir.mkdir(parents=True, exist_ok=True)
    sc = {"total_cases": total, "generation_success_count": success,
          "generation_failure_count": failure, "raw_text_non_null_count": raw_non_null}
    (run_dir / "scorecard.json").write_text(json.dumps(sc), encoding="utf-8")
    (run_dir / "case_results.json").write_text("[]", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")


def test_is_complete_detects_complete(tmp_path: Path):
    run_dir = tmp_path / "cell"
    _write_cell(run_dir)
    assert runner._is_complete(run_dir) is True


def test_is_complete_wrong_total(tmp_path: Path):
    run_dir = tmp_path / "cell"
    _write_cell(run_dir, total=119)
    assert runner._is_complete(run_dir) is False


def test_is_complete_requires_generation_success(tmp_path: Path):
    run_dir = tmp_path / "cell"
    _write_cell(run_dir, success=119)
    assert runner._is_complete(run_dir) is False


def test_is_complete_requires_zero_failures(tmp_path: Path):
    run_dir = tmp_path / "cell"
    _write_cell(run_dir, failure=1)
    assert runner._is_complete(run_dir) is False


def test_is_complete_requires_raw_text_non_null(tmp_path: Path):
    run_dir = tmp_path / "cell"
    _write_cell(run_dir, raw_non_null=119)
    assert runner._is_complete(run_dir) is False


def test_run_manifest_fields():
    runs = _matrix()
    m = runner._run_manifest(runs[0], FROZEN_TEST_SHA256, 120)
    assert m["experiment_class"] == "exploratory_currentdata"
    assert m["formal_release_eligible"] is False
    assert "tool-context trajectories skipped" in m["known_data_limitation"]
    assert m["total_cases"] == 120
    assert m["frozen_test_sha256"] == FROZEN_TEST_SHA256
    # FIX 3 — manifest exposes generation validity counts.
    assert m["generation_success_count"] is None  # populated at write time
    assert m["generation_failure_count"] is None
    assert m["raw_text_non_null_count"] is None


# ── R9: canonical inference reuse + generation fail-closed ───────────────────


def _mk_run(tmp_path: Path, adapter_suffix="a1"):
    """Build a minimal run object with a real adapter dir so adapter gate passes."""
    class _Run:
        run_id = f"phase04_1_7b_sft_u_{adapter_suffix}"
        model_id = "Qwen/Qwen3-1.7B-Base"
        stage = "sft"
        pref_beta = None
        variant = "u"
        use_structured_output = False
        model_key = "1_7b"
        model_size = "1.7B"
        adapter_name_or_path = tmp_path / "adapters" / adapter_suffix
        output_dir = tmp_path / "cells" / f"cell_{adapter_suffix}"

    run = _Run()
    run.adapter_name_or_path.mkdir(parents=True, exist_ok=True)
    (run.adapter_name_or_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    return run


def _ok_gen(case_id: str = "c1") -> GenerationResult:
    return GenerationResult(case_id=case_id, backend_name="fake",
                            raw_text='{"action":"ok","reply_type":"text"}')


class _FakeBackend:
    """Fake Phase04 backend: records whether backend.generate was fed a canonical
    list[Message] (PromptBuilder path) vs a raw BookingRuntimeInput."""

    def __init__(self, gen=None):
        self.calls = []
        self._gen = gen or _ok_gen

    def load(self, config):  # noqa: D401
        pass

    def generate(self, messages, params, case_id=None):
        # Canonical PromptBuilder path feeds backend.generate a list[dict] messages
        # with a "role" key — NOT a raw BookingRuntimeInput object.
        canonical = (
            isinstance(messages, list)
            and all(isinstance(m, dict) and "role" in m for m in messages)
        )
        self.calls.append((case_id, canonical, messages))
        return self._gen(case_id)


def _case(cid: str, input_obj: object):
    class _Case:
        def __init__(self, cid, input_obj):
            self.id = cid
            self.input = input_obj
            self.tags = ["p2"]
            self.output_kind = "action"

    return _Case(cid, input_obj)


def _patch_scorers(monkeypatch):
    """Decouple _score_case from the real Protocol/Task scorers so R9 unit tests
    focus only on the runner's generation validity / provenance behavior."""

    class _Pass:
        passed = True
        details = {}

    class _FakeProtocol:
        def score(self, case, gen):
            return _Pass()

    class _FakeTask:
        def score(self, case, gen, evidence):
            return _Score()

    class _Score:
        score = 1.0
        details = {"structured_score": 1.0, "reply_score": 1.0}

    monkeypatch.setattr(runner, "ProtocolScorer", _FakeProtocol)
    monkeypatch.setattr(runner, "TaskCorrectnessScorer", _FakeTask)
    monkeypatch.setattr(runner, "derive_tool_evidence", lambda case_input: None)


def test_run_cell_uses_canonical_run_inference(monkeypatch, tmp_path: Path):
    """FIX 1: _run_cell must drive backend via run_inference -> PromptBuilder ->
    list[Message]; never pass BookingRuntimeInput directly to generate."""
    import homechef_booking.inference.runner as real_runner

    _patch_scorers(monkeypatch)
    run = _mk_run(tmp_path)
    fake = _FakeBackend()
    monkeypatch.setattr(runner, "Phase04HFTransformersBackend", lambda: fake)
    monkeypatch.setattr(runner, "run_inference", real_runner.run_inference)

    # Record the BookingRuntimeInput that PromptBuilder receives.
    captured = {}

    class _PB:
        def build_messages(self, runtime_input):
            captured["input"] = runtime_input
            return [{"role": "user", "content": "hi"}]

    monkeypatch.setattr(real_runner, "PromptBuilder", _PB)

    runtime_input = object()
    runner._run_cell(run, [_case("c1", runtime_input)], "sha")
    # backend.generate got a list[Message] -> canonical path.
    assert fake.calls and fake.calls[0][1] is True
    # PromptBuilder received the BookingRuntimeInput (never direct to backend).
    assert captured.get("input") is runtime_input
    assert (run.output_dir / "scorecard.json").exists()


def test_run_cell_hard_fails_on_generation_error(monkeypatch, tmp_path: Path):
    """FIX 2: generation error_type is not None -> RuntimeError, no scorecard."""
    run = _mk_run(tmp_path)

    def _err_gen(case_id):
        return GenerationResult(case_id=case_id, backend_name="fake",
                                error_type="generation_failed", error_message="boom")

    fake = _FakeBackend(gen=_err_gen)
    monkeypatch.setattr(runner, "Phase04HFTransformersBackend", lambda: fake)
    monkeypatch.setattr(runner, "run_inference",
                        lambda cid, inp, backend, params: fake.generate([], params, case_id=cid))

    import pytest

    with pytest.raises(RuntimeError, match="generation error"):
        runner._run_cell(run, [_case("c1", object())], "sha")
    assert not (run.output_dir / "scorecard.json").exists()


def test_run_cell_hard_fails_on_empty_raw_text(monkeypatch, tmp_path: Path):
    """FIX 2: raw_text None/empty -> RuntimeError, no scorecard."""
    run = _mk_run(tmp_path, adapter_suffix="a2")

    def _empty_gen(case_id):
        return GenerationResult(case_id=case_id, backend_name="fake", raw_text="   ")

    fake = _FakeBackend(gen=_empty_gen)
    monkeypatch.setattr(runner, "Phase04HFTransformersBackend", lambda: fake)
    monkeypatch.setattr(runner, "run_inference",
                        lambda cid, inp, backend, params: fake.generate([], params, case_id=cid))

    import pytest

    with pytest.raises(RuntimeError, match="empty raw_text"):
        runner._run_cell(run, [_case("c1", object())], "sha")
    assert not (run.output_dir / "scorecard.json").exists()


def test_aggregate_reports_generation_validity_counts():
    results = [
        {"protocol_pass": True, "task_correctness": 1.0, "structured_score": 1.0,
         "reply_score": 1.0, "valid_json": True, "effective_pass": True,
         "latency_ms": 10, "ttft_ms": 5, "tokens_per_second": 3.0,
         "primary_failure": None, "generation_error_type": None,
         "generation_error_message": None, "raw_text": '{"a":1}'},
        {"protocol_pass": False, "task_correctness": 0.0, "structured_score": 0.0,
         "reply_score": 0.0, "valid_json": False, "effective_pass": False,
         "latency_ms": None, "ttft_ms": None, "tokens_per_second": None,
         "primary_failure": "boom", "generation_error_type": None,
         "generation_error_message": None, "raw_text": "x"},
    ]
    agg = runner._aggregate(results)
    assert agg["total_cases"] == 2
    assert agg["generation_success_count"] == 2
    assert agg["generation_failure_count"] == 0
    assert agg["raw_text_non_null_count"] == 2


def test_score_case_records_null_generation_error_on_success(monkeypatch):
    # Reuse runner._score_case with a successful GenerationResult: error fields null.
    from homechef_booking.inference.response import GenerationResult

    _patch_scorers(monkeypatch)
    gen = GenerationResult(case_id="c1", backend_name="hf", raw_text='{"action":"x"}')
    res = runner._score_case(_case("c1", None), gen)
    assert res["generation_error_type"] is None
    assert res["generation_error_message"] is None
    assert res["raw_text"] == '{"action":"x"}'


# ── Summarizer: 16-cell comparison (synthetic scorecards) ────────────────────


def _mk_cell(run_id: str, **over):
    base = {"run_id": run_id, "model": "1_7b", "model_size": "1.7B", "stage": "sft",
            "beta": None, "variant": "u", "available": True,
            "protocol_pass_rate": 0.5, "effective_pass_rate": 0.3,
            "mean_task_correctness": 0.6, "critical_error_rate": 0.2,
            "p95_latency_s": 1.0, "mean_ttft_s": 0.1, "mean_tokens_per_second": 10.0}
    base.update(over)
    return base


def _mk_scorecard_json(protocol=0.8, task=0.9, eff=0.7, crit=0.05):
    return {"protocol_pass_rate": protocol, "mean_task_correctness": task,
            "effective_pass_rate": eff, "critical_error_rate": crit,
            "total_cases": 120,
            "performance": {"p95_latency_s": 0.5, "mean_ttft_s": 0.05,
                            "mean_tokens_per_second": 20.0}}


def test_summarizer_16_cells_with_synthetic(tmp_path: Path):
    # Create 12 synthetic Phase04 scorecards + 4 synthetic Phase02 base.
    out = tmp_path / "out"
    phase02 = tmp_path / "phase02"
    for rid in EXPECTED_IDS:
        d = out / rid
        d.mkdir(parents=True)
        (d / "scorecard.json").write_text(json.dumps(_mk_scorecard_json()), encoding="utf-8")
    for sub in ("qwen3_1_7b_unstructured", "qwen3_1_7b_structured",
                "qwen3_4b_unstructured", "qwen3_4b_structured"):
        d = phase02 / sub
        d.mkdir(parents=True)
        (d / "aggregate.json").write_text(json.dumps(_mk_scorecard_json(protocol=0.3, task=0.4)), encoding="utf-8")

    cells = summer.build_comparison(out, phase02)
    assert len(cells) == 16
    assert sum(1 for c in cells if c.get("available")) == 16
    deltas = summer.build_deltas(cells)
    assert set(deltas) == {"1_7b", "4b"}
    selection = summer.select_exploratory_best(cells)
    assert selection["exploratory_best_run"] is not None
    assert selection["formal_release_eligible"] is False


def test_summarizer_16cell_comparison_file(tmp_path: Path):
    out = tmp_path / "out"
    phase02 = tmp_path / "phase02"
    for rid in EXPECTED_IDS:
        d = out / rid
        d.mkdir(parents=True)
        (d / "scorecard.json").write_text(json.dumps(_mk_scorecard_json()), encoding="utf-8")
    for sub in ("qwen3_1_7b_unstructured", "qwen3_1_7b_structured",
                "qwen3_4b_unstructured", "qwen3_4b_structured"):
        d = phase02 / sub
        d.mkdir(parents=True)
        (d / "aggregate.json").write_text(json.dumps(_mk_scorecard_json()), encoding="utf-8")
    cells = summer.build_comparison(out, phase02)
    deltas = summer.build_deltas(cells)
    summer.write_reports(out, cells, deltas, {}, {"exploratory_best_run": None})
    assert (out / "phase04_16cell_comparison.json").exists()
    assert (out / "phase04_16cell_comparison.csv").exists()
    assert (out / "phase04_16cell_comparison.md").exists()
    assert (out / "phase04_deltas.json").exists()


def test_case_diff_synthetic(tmp_path: Path):
    parent_dir = tmp_path / "phase04_1_7b_sft_u"
    child_dir = tmp_path / "phase04_1_7b_dpo_b01_u"
    parent_dir.mkdir(parents=True)
    child_dir.mkdir(parents=True)
    (parent_dir / "case_results.json").write_text(json.dumps([
        {"case_id": "c1", "effective_pass": False},
        {"case_id": "c2", "effective_pass": True},
    ]), encoding="utf-8")
    (child_dir / "case_results.json").write_text(json.dumps([
        {"case_id": "c1", "effective_pass": True},
        {"case_id": "c2", "effective_pass": False},
    ]), encoding="utf-8")
    diff = summer.build_case_diff(tmp_path)
    key = "1_7b_b01_u"
    assert diff[key]["improved_cases"] == ["c1"]
    assert diff[key]["regressed_cases"] == ["c2"]
    assert diff[key]["net_effective_pass"] == 0
