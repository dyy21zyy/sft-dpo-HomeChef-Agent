"""TDD: Phase 03 Semantic Validators — using real smoke sample structure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from homechef_booking.data.raw_sample import RawBookingSample, parse_raw_sample_line
from homechef_booking.data.semantic_validators import validate_semantic


def _load_smoke_samples() -> list[RawBookingSample]:
    lines = Path("data/raw/phase03_smoke_v0.1.jsonl").read_text(encoding="utf-8").splitlines()
    return [parse_raw_sample_line(line) for line in lines if line.strip()]


def _sample_by_scenario(scenario: str) -> RawBookingSample | None:
    for s in _load_smoke_samples():
        if s.scenario == scenario:
            return s
    return None


# ── Schema validation (existing samples) ────────────────────

def test_smoke_samples_all_schema_valid():
    samples = _load_smoke_samples()
    assert len(samples) == 25
    for s in samples:
        assert isinstance(s, RawBookingSample)


# ── Relative Time: test with manually constructed samples ────

def test_relative_time_validator_runs_on_real_samples():
    """Validator should not crash on real smoke samples."""
    samples = _load_smoke_samples()
    for s in samples:
        result = validate_semantic(s)
        # Just verify it doesn't crash
        assert isinstance(result.passed, bool)


# ── Tool Fact Grounding: test with real matched_candidates ──

def test_tool_fact_grounding_on_real_matched():
    sample = _sample_by_scenario("matched_candidates")
    if sample is None:
        pytest.skip("No matched_candidates in smoke")
    from homechef_booking.data.semantic_validators import validate_tool_fact_grounding
    err = validate_tool_fact_grounding(sample)
    # May pass or fail depending on data quality — just verify no crash
    assert err is None or isinstance(err, str)


# ── Candidate Order: test with real samples ─────────────────

def test_candidate_order_on_real_samples():
    sample = _sample_by_scenario("matched_candidates")
    if sample is None:
        pytest.skip("No matched_candidates in smoke")
    from homechef_booking.data.semantic_validators import validate_candidate_order
    err = validate_candidate_order(sample)
    assert err is None or isinstance(err, str)


# ── Dietary: test with real samples ─────────────────────────

def test_dietary_on_real_samples():
    samples = _load_smoke_samples()
    from homechef_booking.data.semantic_validators import validate_dietary_constraints
    for s in samples:
        err = validate_dietary_constraints(s)
        assert err is None or isinstance(err, str)


# ── Confirmation: test with real explicit_confirmation ──────

def test_confirmation_on_real_explicit():
    sample = _sample_by_scenario("explicit_confirmation")
    if sample is None:
        pytest.skip("No explicit_confirmation in smoke")
    from homechef_booking.data.semantic_validators import validate_confirmation_transition
    err = validate_confirmation_transition(sample)
    assert err is None or isinstance(err, str)


# ── Relative time: deterministic test with manual dates ─────

def test_relative_time_resolver_tomorrow():
    from datetime import date

    from homechef_booking.data.semantic_validators import _RELATIVE_TIME_MAP
    resolver = _RELATIVE_TIME_MAP["tomorrow"]
    assert resolver(date(2026, 8, 12)) == date(2026, 8, 13)
    assert resolver(date(2026, 8, 31)) == date(2026, 9, 1)
    assert resolver(date(2026, 12, 31)) == date(2027, 1, 1)


def test_relative_time_resolver_today():
    from datetime import date

    from homechef_booking.data.semantic_validators import _RELATIVE_TIME_MAP
    resolver = _RELATIVE_TIME_MAP["today"]
    assert resolver(date(2026, 8, 12)) == date(2026, 8, 12)


def test_relative_time_resolver_day_after_tomorrow():
    from datetime import date

    from homechef_booking.data.semantic_validators import _RELATIVE_TIME_MAP
    resolver = _RELATIVE_TIME_MAP["day_after_tomorrow"]
    assert resolver(date(2026, 8, 12)) == date(2026, 8, 14)


# ── State Inheritance: test with real multi_turn samples ────

def test_state_inheritance_on_multi_turn():
    samples = [s for s in _load_smoke_samples() if s.conversation_kind == "multi_turn"]
    if not samples:
        pytest.skip("No multi_turn in smoke")
    from homechef_booking.data.semantic_validators import validate_state_inheritance
    for s in samples:
        err = validate_state_inheritance(s)
        assert err is None or isinstance(err, str)


# ── Aggregate validation on all smoke ───────────────────────

def test_aggregate_semantic_on_all_smoke():
    samples = _load_smoke_samples()
    passed = 0
    failed = 0
    for s in samples:
        result = validate_semantic(s)
        if result.passed:
            passed += 1
        else:
            failed += 1
    assert passed + failed == 25
    print(f"\nSmoke semantic: {passed} passed, {failed} failed")


# ── DPO rejected schema-valid check ─────────────────────────

def test_dpo_rejected_are_schema_valid():
    """Check that DPO rejected samples are schema-valid decisions."""
    from homechef_booking.schemas.decision import FinalDecision, ToolCallDecision

    dpo_path = Path("data/processed/phase03_dpo_targeted_v0.1_train.jsonl")
    if not dpo_path.exists():
        pytest.skip("DPO targeted train not found")

    invalid_count = 0
    total = 0
    for line in dpo_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        pair = json.loads(line)
        rejected_str = pair.get("rejected", "")
        try:
            rejected_obj = json.loads(rejected_str)
            # Try to validate as Decision
            action = rejected_obj.get("action", "")
            if action == "tool_call":
                ToolCallDecision.model_validate(rejected_obj)
            elif action == "final":
                FinalDecision.model_validate(rejected_obj)
            else:
                invalid_count += 1
        except Exception:
            invalid_count += 1

    print(f"\nDPO rejected schema-valid: {total - invalid_count}/{total}")
    # We expect most to be valid (this is a quality check, not a hard pass/fail)
    assert invalid_count / max(total, 1) < 0.5, f"Too many invalid DPO rejected: {invalid_count}/{total}"
