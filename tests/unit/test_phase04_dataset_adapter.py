"""Phase 04 Task 2 — Dataset adapter and training preflight tests."""

import json
from pathlib import Path

import pytest

from homechef_booking.training.dataset_adapter import (
    DatasetPreflightReport,
    validate_dpo_for_training,
    validate_sft_for_training,
)


# ── Accepted Phase 03 counts ────────────────────────────────────────────────


def test_sft_train_file_matches_accepted_phase03_count():
    report = validate_sft_for_training(
        Path("data/processed/phase03_sft_v0.1_train.jsonl"), expected_count=540
    )
    assert report.total_rows == 540
    assert report.error_count == 0
    assert report.passed


def test_sft_val_file_matches_accepted_phase03_count():
    report = validate_sft_for_training(
        Path("data/processed/phase03_sft_v0.1_val.jsonl"), expected_count=60
    )
    assert report.total_rows == 60
    assert report.error_count == 0
    assert report.passed


def test_targeted_dpo_train_file_matches_accepted_phase03_count():
    report = validate_dpo_for_training(
        Path("data/processed/phase03_dpo_targeted_v0.1_train.jsonl"), expected_count=216
    )
    assert report.total_rows == 216
    assert report.error_count == 0
    assert report.passed


def test_targeted_dpo_val_file_matches_accepted_phase03_count():
    report = validate_dpo_for_training(
        Path("data/processed/phase03_dpo_targeted_v0.1_val.jsonl"), expected_count=24
    )
    assert report.total_rows == 24
    assert report.error_count == 0
    assert report.passed


# ── Rejection of eval suites as training inputs ─────────────────────────────


def test_preflight_rejects_frozen_test_as_training_input():
    report = validate_sft_for_training(
        Path("data/eval/frozen_test.jsonl"), expected_count=120
    )
    assert not report.passed
    assert any("forbidden" in e.lower() or "frozen" in e.lower() for e in report.errors)


def test_preflight_rejects_diagnostic_dev_as_training_input():
    report = validate_sft_for_training(
        Path("data/dev/diagnostic_dev.jsonl"), expected_count=80
    )
    assert not report.passed
    assert any("forbidden" in e.lower() or "diagnostic" in e.lower() for e in report.errors)


def test_preflight_rejects_phase02_benchmark_output():
    report = validate_sft_for_training(
        Path("reports/generated/phase02/mock_results.jsonl"), expected_count=1
    )
    assert not report.passed
    assert any("phase02" in e.lower() or "benchmark" in e.lower() for e in report.errors)


# ── Count mismatch detection ────────────────────────────────────────────────


def test_preflight_detects_count_mismatch():
    report = validate_sft_for_training(
        Path("data/processed/phase03_sft_v0.1_train.jsonl"), expected_count=999
    )
    assert not report.passed
    assert any("count" in e.lower() or "expected 999" in e.lower() or "540" in e for e in report.errors)


# ── Row-level forbidden ID checks ───────────────────────────────────────────


def test_preflight_rejects_row_with_frozen_id():
    # Create a temp file with a row containing a forbidden ID prefix
    bad_rows = [json.dumps({
        "id": "frozen-test-row",
        "raw_id": "r1",
        "dataset_version": "v0.1",
        "messages": [{"role": "user", "content": "hello"}],
        "prompt_sha256": "a" * 64,
        "completion_sha256": "b" * 64,
        "tags": [],
    })]
    tmp = Path("data/processed/_test_frozen_row.jsonl")
    tmp.write_text("\n".join(bad_rows), encoding="utf-8")
    try:
        report = validate_sft_for_training(tmp, expected_count=1)
        assert not report.passed
        assert any("forbidden" in e.lower() or "frozen" in e.lower() for e in report.errors)
    finally:
        tmp.unlink(missing_ok=True)


def test_preflight_rejects_row_with_diagnostic_id():
    bad_rows = [json.dumps({
        "id": "diagnostic-dev-row",
        "raw_id": "r1",
        "dataset_version": "v0.1",
        "messages": [{"role": "user", "content": "hello"}],
        "prompt_sha256": "a" * 64,
        "completion_sha256": "b" * 64,
        "tags": [],
    })]
    tmp = Path("data/processed/_test_diag_row.jsonl")
    tmp.write_text("\n".join(bad_rows), encoding="utf-8")
    try:
        report = validate_sft_for_training(tmp, expected_count=1)
        assert not report.passed
        assert any("forbidden" in e.lower() or "diagnostic" in e.lower() for e in report.errors)
    finally:
        tmp.unlink(missing_ok=True)


# ── DatasetPreflightReport model ─────────────────────────────────────────────


def test_preflight_report_pydantic_model():
    report = DatasetPreflightReport(
        dataset_type="sft",
        total_rows=540,
        expected_count=540,
        passed=True,
        errors=[],
    )
    assert report.passed
    assert report.total_rows == 540
    assert report.expected_count == 540
    assert report.dataset_type == "sft"
