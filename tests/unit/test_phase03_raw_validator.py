"""Task 2 Phase 03: Raw validator smoke tests."""

from pathlib import Path

from homechef_booking.data.raw_validator import validate_raw_jsonl


def test_raw_smoke_candidate_fixture_validates():
    report = validate_raw_jsonl(Path("tests/fixtures/datasets/raw_smoke_candidate.jsonl"))
    assert report.total == 2
    assert report.error_count == 0


def test_raw_invalid_candidate_fixture_reports_errors():
    report = validate_raw_jsonl(Path("tests/fixtures/datasets/raw_invalid_candidate.jsonl"))
    assert report.total == 2
    assert report.error_count == 2
    assert any("output_kind" in error.message or "contract" in error.message for error in report.errors)
