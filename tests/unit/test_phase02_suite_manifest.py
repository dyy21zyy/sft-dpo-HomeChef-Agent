"""Task 1 Phase 02: Suite manifest and integrity validation tests."""

import json
from pathlib import Path

import pytest

from homechef_booking.evaluation.suite_manifest import (
    SuiteManifest,
    compute_sha256,
    load_suite_manifest,
    validate_suite_manifest,
)

FIXTURE_DIR = Path("tests/fixtures/evaluation")


def _make_minimal_jsonl(tmp_path: Path, case_count: int) -> Path:
    """Create a minimal valid EvalCase JSONL file with `case_count` lines."""
    lines = []
    for i in range(1, case_count + 1):
        lines.append(json.dumps({
            "id": f"minimal_case_{i:03d}",
            "output_kind": "final",
            "conversation_kind": "single_turn",
            "input": {
                "history": [],
                "current_state": {
                    "booking_state": {
                        "service_date": None, "start_time": None, "people": None, "address": None,
                        "cuisine": None, "budget_min": None, "budget_max": None, "menu": [],
                        "chef_id": None, "chef_name": None, "ingredient_purchase": None,
                        "dietary_constraints": [], "occasion": None, "confirmation": None,
                    },
                    "chef_query_status": "not_checked",
                    "candidate_chefs": [],
                    "awaiting_confirmation": False,
                },
                "user_input": "你好",
                "current_time": "2026-08-09 18:00",
                "available_tools": [],
            },
            "expected": {
                "action": "final",
                "booking_state": {
                    "service_date": None, "start_time": None, "people": None, "address": None,
                    "cuisine": None, "budget_min": None, "budget_max": None, "menu": [],
                    "chef_id": None, "chef_name": None, "ingredient_purchase": None,
                    "dietary_constraints": [], "occasion": None, "confirmation": None,
                },
                "chef_query_status": "not_checked",
                "candidate_chefs": [],
                "info_complete": False,
                "unrelated": False,
                "missing_info": ["service_date"],
                "reply_type": "ask_service_date",
                "reply": "请提供服务日期",
            },
            "assertions": [],
            "tags": [],
            "reply_expectations": {},
        }, ensure_ascii=False, sort_keys=True))
    path = tmp_path / "minimal.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_manifest_validation_fails_when_case_count_differs(tmp_path: Path):
    jsonl_path = _make_minimal_jsonl(tmp_path, 5)
    sha = compute_sha256(jsonl_path)
    manifest = SuiteManifest(
        suite_id="test_v1",
        contract_id="homechef-booking-v1",
        case_file="test.jsonl",
        case_count=3,
        sha256=sha,
        frozen=True,
        approved_by="test",
        notes="test",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="case_count"):
        validate_suite_manifest(manifest_path, root=tmp_path, resolve_case_file=jsonl_path)


def test_manifest_validation_fails_when_sha256_differs(tmp_path: Path):
    jsonl_path = _make_minimal_jsonl(tmp_path, 5)
    manifest = SuiteManifest(
        suite_id="test_v1",
        contract_id="homechef-booking-v1",
        case_file="test.jsonl",
        case_count=5,
        sha256="0000000000000000000000000000000000000000000000000000000000000000",
        frozen=True,
        approved_by="test",
        notes="test",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256"):
        validate_suite_manifest(manifest_path, root=tmp_path, resolve_case_file=jsonl_path)


def test_manifest_requires_frozen_true_for_frozen_suites(tmp_path: Path):
    jsonl_path = _make_minimal_jsonl(tmp_path, 5)
    sha = compute_sha256(jsonl_path)
    manifest = SuiteManifest(
        suite_id="test_v1",
        contract_id="homechef-booking-v1",
        case_file="test.jsonl",
        case_count=5,
        sha256=sha,
        frozen=False,
        approved_by="test",
        notes="test",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen"):
        validate_suite_manifest(manifest_path, root=tmp_path, resolve_case_file=jsonl_path, require_frozen=True)


def test_manifest_validation_passes_with_correct_data(tmp_path: Path):
    jsonl_path = _make_minimal_jsonl(tmp_path, 5)
    sha = compute_sha256(jsonl_path)
    manifest = SuiteManifest(
        suite_id="test_v1",
        contract_id="homechef-booking-v1",
        case_file="test.jsonl",
        case_count=5,
        sha256=sha,
        frozen=True,
        approved_by="test",
        notes="test",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")

    result = validate_suite_manifest(manifest_path, root=tmp_path, resolve_case_file=jsonl_path)
    assert result.suite_id == "test_v1"
    assert result.case_count == 5


def test_load_suite_manifest_parses_valid_manifest(tmp_path: Path):
    manifest_data = {
        "suite_id": "test_v1",
        "contract_id": "homechef-booking-v1",
        "case_file": "data/eval/test.jsonl",
        "case_count": 100,
        "sha256": "a" * 64,
        "frozen": True,
        "approved_by": "test",
        "notes": "test suite",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, ensure_ascii=False), encoding="utf-8")

    manifest = load_suite_manifest(manifest_path)
    assert manifest.suite_id == "test_v1"
    assert manifest.frozen is True
    assert manifest.case_count == 100
