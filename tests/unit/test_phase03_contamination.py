"""Task 4 Phase 03: Contamination check tests."""

from pathlib import Path

from homechef_booking.data.contamination import check_raw_contamination


def test_overlap_probe_detects_frozen_eval_overlap():
    report = check_raw_contamination(
        Path("tests/fixtures/datasets/frozen_overlap_probe.jsonl"),
        [Path("data/eval/frozen_test.jsonl")],
    )
    assert report.overlap_count == 1


def test_smoke_candidate_has_zero_frozen_and_diagnostic_overlap():
    report = check_raw_contamination(
        Path("tests/fixtures/datasets/raw_smoke_candidate.jsonl"),
        [Path("data/eval/frozen_test.jsonl"), Path("data/dev/diagnostic_dev.jsonl")],
    )
    assert report.overlap_count == 0
