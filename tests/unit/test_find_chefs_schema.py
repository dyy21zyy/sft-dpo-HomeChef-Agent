import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from homechef_booking.schemas.tools import FindChefsInput, parse_find_chefs_result

FIXTURES = (
    Path(__file__).resolve().parent.parent.parent
    / "tests"
    / "fixtures"
    / "contracts"
    / "valid"
)


def load_tool_result_fixture(mode: str, status: str) -> dict:
    fixture_map = {
        ("search", "matched"): "find_chefs_search_matched_result.json",
        ("search", "no_match"): "find_chefs_search_no_match_result.json",
        ("search", "out_of_service_area"): "find_chefs_search_out_of_service_area_result.json",
        ("search", "error"): "find_chefs_search_error_result.json",
        ("specific", "available"): "find_chefs_specific_available_result.json",
        ("specific", "unavailable"): "find_chefs_specific_unavailable_result.json",
        ("specific", "not_found"): "find_chefs_specific_not_found_result.json",
        ("specific", "out_of_service_area"): "find_chefs_specific_out_of_service_area_result.json",
        ("specific", "error"): "find_chefs_specific_error_result.json",
    }
    filename = fixture_map[(mode, status)]
    return json.loads((FIXTURES / filename).read_text(encoding="utf-8"))


def test_find_chefs_input_has_exact_contract_arguments() -> None:
    value = FindChefsInput.model_validate({
        "chef_name": None,
        "service_date": "2026-08-15",
        "start_time": "18:00",
        "people": 6,
        "address": "杨浦",
        "cuisine": "川菜",
        "budget_min": 800.0,
        "budget_max": 1200.0,
        "menu": [],
        "ingredient_purchase": None,
        "dietary_constraints": [],
        "occasion": "家庭聚餐",
    })
    assert list(value.model_dump().keys()) == [
        "chef_name", "service_date", "start_time", "people", "address", "cuisine",
        "budget_min", "budget_max", "menu", "ingredient_purchase",
        "dietary_constraints", "occasion",
    ]


def test_matched_result_preserves_candidate_order() -> None:
    result = parse_find_chefs_result({
        "mode": "search",
        "status": "matched",
        "candidates": [
            {"chef_id": "C003", "chef_name": "张伟"},
            {"chef_id": "C007", "chef_name": "李明"},
        ],
    })
    assert [c.chef_id for c in result.candidates] == ["C003", "C007"]


@pytest.mark.parametrize("mode,status", [
    ("search", "matched"),
    ("search", "no_match"),
    ("search", "out_of_service_area"),
    ("search", "error"),
    ("specific", "available"),
    ("specific", "unavailable"),
    ("specific", "not_found"),
    ("specific", "out_of_service_area"),
    ("specific", "error"),
])
def test_all_find_chefs_mode_status_pairs_have_explicit_fixtures(mode: str, status: str) -> None:
    fixture = load_tool_result_fixture(mode, status)
    assert parse_find_chefs_result(fixture).status == status


def test_unavailable_alternatives_preserve_order() -> None:
    result = parse_find_chefs_result({
        "mode": "specific",
        "status": "unavailable",
        "requested_chef": "张伟",
        "alternatives": [
            {"chef_id": "C005", "chef_name": "王芳"},
            {"chef_id": "C009", "chef_name": "赵强"},
        ],
    })
    assert [c.chef_id for c in result.alternatives] == ["C005", "C009"]


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_find_chefs_result({
            "mode": "search",
            "status": "matched",
            "candidates": [{"chef_id": "C003", "chef_name": "张伟"}],
            "extra_field": "forbidden",
        })
