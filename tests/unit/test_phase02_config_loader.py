"""Task 3 Phase 02: Suite configs and case loader smoke tests."""

import json
from pathlib import Path

from homechef_booking.evaluation.sample import load_eval_cases
from homechef_booking.evaluation.suite_manifest import validate_suite_manifest

FROZEN_CASES = Path("data/eval/frozen_test.jsonl")
FROZEN_MANIFEST = Path("data/eval/frozen_test.manifest.json")
DIAG_CASES = Path("data/dev/diagnostic_dev.jsonl")
DIAG_MANIFEST = Path("data/dev/diagnostic_dev.manifest.json")


def test_frozen_manifest_passes_integrity_check():
    result = validate_suite_manifest(FROZEN_MANIFEST, root=Path("."))
    assert result.suite_id == "phase02_frozen_test_v1"
    assert result.case_count == 120
    assert result.frozen is True


def test_diagnostic_manifest_passes_integrity_check():
    result = validate_suite_manifest(DIAG_MANIFEST, root=Path("."))
    assert result.suite_id == "phase02_diagnostic_dev_v1"
    assert result.case_count == 80
    assert result.frozen is False


def test_frozen_cases_pass_full_load_eval_cases():
    cases = load_eval_cases(FROZEN_CASES)
    assert len(cases) == 120
    for case in cases:
        assert case.output_kind == case.expected.action


def test_diagnostic_cases_pass_full_load_eval_cases():
    cases = load_eval_cases(DIAG_CASES)
    assert len(cases) == 80
    for case in cases:
        assert case.output_kind == case.expected.action


def test_frozen_cases_manifest_sha256_matches_file():
    from homechef_booking.evaluation.suite_manifest import compute_sha256, load_suite_manifest
    manifest = load_suite_manifest(FROZEN_MANIFEST)
    actual = compute_sha256(FROZEN_CASES)
    assert manifest.sha256 == actual


def test_diagnostic_cases_manifest_sha256_matches_file():
    from homechef_booking.evaluation.suite_manifest import compute_sha256, load_suite_manifest
    manifest = load_suite_manifest(DIAG_MANIFEST)
    actual = compute_sha256(DIAG_CASES)
    assert manifest.sha256 == actual
