"""Tests for Fix 7: complete business invariants."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from homechef_booking.schemas.booking import (
    BookingSlot,
    is_affirmative,
    missing_required_slots,
)
from homechef_booking.schemas.decision import (
    ToolCallDecision,
)
from homechef_booking.schemas.tools import FindChefsInput
from homechef_booking.validation.contract_validator import (
    validate_decision,
    validate_runtime_input,
)

VALID = Path("tests/fixtures/contracts/valid")
INVALID = Path("tests/fixtures/contracts/invalid")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# Fix 1: coercion negative tests


@pytest.mark.parametrize(
    "field,value",
    [
        ("people", "6"),
        ("people", 6.0),
        ("people", True),
        ("ingredient_purchase", 1),
        ("ingredient_purchase", "true"),
        ("budget_min", "800"),
        ("budget_max", True),
    ],
)
def test_strict_types_reject_coercion(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        BookingSlot(**{field: value})


# Fix 2: ToolCallDecision.arguments is FindChefsInput


def test_tool_call_arguments_must_have_exactly_12_keys() -> None:
    decision = ToolCallDecision.model_validate(
        load(VALID / "tool_call_decision.json")
    )
    assert len(decision.arguments.model_dump()) == 12


def test_tool_call_arguments_missing_key_fails() -> None:
    with pytest.raises(ValidationError):
        FindChefsInput.model_validate({"chef_name": None})


def test_tool_call_arguments_extra_key_fails() -> None:
    with pytest.raises(ValidationError):
        FindChefsInput.model_validate({
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
            "extra_field": "forbidden",
        })


def test_tool_call_arguments_wrong_type_fails() -> None:
    with pytest.raises(ValidationError):
        FindChefsInput.model_validate({
            "chef_name": None,
            "service_date": "2026-08-15",
            "start_time": "18:00",
            "people": "6",
            "address": "杨浦",
            "cuisine": "川菜",
            "budget_min": 800.0,
            "budget_max": 1200.0,
            "menu": [],
            "ingredient_purchase": None,
            "dietary_constraints": [],
            "occasion": "家庭聚餐",
        })


# Fix 7: Business invariants


def test_tool_call_only_when_required_slots_complete() -> None:
    runtime = load(VALID / "minimal_runtime_input.json")
    runtime["current_state"]["booking_state"]["service_date"] = None
    decision = load(VALID / "tool_call_decision.json")
    issues = validate_decision(decision, runtime)
    assert any("required slots" in i.message for i in issues)


def test_info_complete_consistency() -> None:
    decision = load(VALID / "final_missing_address.json")
    decision["info_complete"] = True
    issues = validate_decision(decision)
    assert any("info_complete" in i.path for i in issues)


def test_missing_info_canonical_order() -> None:
    slot = BookingSlot()
    assert missing_required_slots(slot) == [
        "service_date",
        "start_time",
        "people",
        "address",
    ]


def test_unrelated_handoff_consistency() -> None:
    decision = load(INVALID / "final_unrelated_not_handoff.json")
    issues = validate_decision(decision)
    assert any("unrelated" in i.path for i in issues)


def test_state_invalidation_on_dependency_mutation() -> None:
    runtime = load(VALID / "history_tool_continuation.json")
    decision = load(INVALID / "state_dependency_mutation_stale_tool_facts.json")
    issues = validate_decision(decision, runtime)
    assert any("dependency mutation" in i.message for i in issues)


def test_candidate_order_preservation() -> None:
    runtime = load(VALID / "history_tool_continuation.json")
    decision = load(INVALID / "final_reordered_candidates.json")
    issues = validate_decision(decision, runtime)
    assert any("candidate order" in i.message for i in issues)


def test_candidate_selection_provenance() -> None:
    runtime = load(VALID / "history_tool_continuation.json")
    decision = load(INVALID / "final_fabricated_chef.json")
    issues = validate_decision(decision, runtime)
    assert any("not found in tool candidates" in i.message for i in issues)


def test_booking_authorized_requires_awaiting_confirmation() -> None:
    runtime = load(VALID / "minimal_runtime_input.json")
    runtime["current_state"]["awaiting_confirmation"] = False
    decision = {
        "action": "final",
        "booking_state": {
            "service_date": "2026-08-15",
            "start_time": "18:00",
            "people": 6,
            "address": "杨浦",
            "cuisine": "川菜",
            "budget_min": 800.0,
            "budget_max": 1200.0,
            "menu": [],
            "chef_id": "C003",
            "chef_name": "张伟",
            "ingredient_purchase": None,
            "dietary_constraints": [],
            "occasion": "家庭聚餐",
            "confirmation": True,
        },
        "chef_query_status": "available",
        "candidate_chefs": [],
        "info_complete": True,
        "unrelated": False,
        "missing_info": [],
        "reply_type": "booking_authorized",
        "reply": "ok",
    }
    issues = validate_decision(decision, runtime)
    assert any("awaiting_confirmation" in i.message for i in issues)


def test_booking_authorized_requires_verified_chef() -> None:
    runtime = load(VALID / "minimal_runtime_input.json")
    runtime["current_state"]["awaiting_confirmation"] = True
    decision = {
        "action": "final",
        "booking_state": {
            "service_date": "2026-08-15",
            "start_time": "18:00",
            "people": 6,
            "address": "杨浦",
            "cuisine": None,
            "budget_min": None,
            "budget_max": None,
            "menu": [],
            "chef_id": None,
            "chef_name": None,
            "ingredient_purchase": None,
            "dietary_constraints": [],
            "occasion": None,
            "confirmation": True,
        },
        "chef_query_status": "not_checked",
        "candidate_chefs": [],
        "info_complete": True,
        "unrelated": False,
        "missing_info": [],
        "reply_type": "booking_authorized",
        "reply": "ok",
    }
    issues = validate_decision(decision, runtime)
    assert any("chef_id" in i.path or "verified" in i.message for i in issues)


def test_deterministic_affirmative_allowlist() -> None:
    assert is_affirmative("确认") is True
    assert is_affirmative("好的") is True
    assert is_affirmative("可以") is True
    assert is_affirmative("嗯，就这样") is False
    assert is_affirmative("不确认") is False


def test_mutation_dominates_confirmation() -> None:
    runtime = load(VALID / "history_tool_continuation.json")
    runtime["current_state"]["awaiting_confirmation"] = True
    runtime["current_state"]["booking_state"]["service_date"] = "2026-08-15"
    runtime["current_state"]["booking_state"]["chef_id"] = "C003"
    runtime["current_state"]["chef_query_status"] = "available"
    runtime["user_input"] = "确认"
    decision = {
        "action": "final",
        "booking_state": {
            "service_date": "2026-08-22",
            "start_time": "18:00",
            "people": 6,
            "address": "杨浦",
            "cuisine": "川菜",
            "budget_min": 800.0,
            "budget_max": 1200.0,
            "menu": [],
            "chef_id": "C003",
            "chef_name": "张伟",
            "ingredient_purchase": None,
            "dietary_constraints": [],
            "occasion": "家庭聚餐",
            "confirmation": True,
        },
        "chef_query_status": "available",
        "candidate_chefs": [],
        "info_complete": True,
        "unrelated": False,
        "missing_info": [],
        "reply_type": "booking_authorized",
        "reply": "ok",
    }
    issues = validate_decision(decision, runtime)
    assert any("dependency mutation" in i.message for i in issues)


def test_matched_candidate_selection_preserves_status() -> None:
    runtime = load(VALID / "history_tool_continuation.json")
    decision = {
        "action": "final",
        "booking_state": {
            "service_date": "2026-08-15",
            "start_time": "18:00",
            "people": 6,
            "address": "杨浦",
            "cuisine": "川菜",
            "budget_min": 800.0,
            "budget_max": 1200.0,
            "menu": [],
            "chef_id": "C003",
            "chef_name": "张伟",
            "ingredient_purchase": None,
            "dietary_constraints": [],
            "occasion": "家庭聚餐",
            "confirmation": False,
        },
        "chef_query_status": "available",
        "candidate_chefs": [
            {"chef_id": "C003", "chef_name": "张伟"},
            {"chef_id": "C007", "chef_name": "李明"},
        ],
        "info_complete": True,
        "unrelated": False,
        "missing_info": [],
        "reply_type": "confirm_specific_chef",
        "reply": "ok",
    }
    issues = validate_decision(decision, runtime)
    assert any("matched" in i.message and "preserve" in i.message for i in issues)


# Fix 4: History tool arguments validation


def test_history_tool_call_arguments_empty_json_fails() -> None:
    runtime = load(VALID / "history_tool_continuation.json")
    runtime["history"][0]["tool_calls"][0]["function"]["arguments"] = "{}"
    issues = validate_runtime_input(runtime)
    assert any("invalid tool_call arguments" in i.message for i in issues)


def test_history_tool_call_arguments_missing_key_fails() -> None:
    runtime = load(VALID / "history_tool_continuation.json")
    runtime["history"][0]["tool_calls"][0]["function"]["arguments"] = (
        json.dumps({"chef_name": None})
    )
    issues = validate_runtime_input(runtime)
    assert any("invalid tool_call arguments" in i.message for i in issues)


def test_history_tool_call_arguments_extra_key_fails() -> None:
    runtime = load(VALID / "history_tool_continuation.json")
    args = json.loads(
        runtime["history"][0]["tool_calls"][0]["function"]["arguments"]
    )
    args["extra_field"] = "forbidden"
    runtime["history"][0]["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
    issues = validate_runtime_input(runtime)
    assert any("invalid tool_call arguments" in i.message for i in issues)


def test_history_tool_call_arguments_wrong_type_fails() -> None:
    runtime = load(VALID / "history_tool_continuation.json")
    args = json.loads(
        runtime["history"][0]["tool_calls"][0]["function"]["arguments"]
    )
    args["people"] = "6"
    runtime["history"][0]["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
    issues = validate_runtime_input(runtime)
    assert any("invalid tool_call arguments" in i.message for i in issues)


def test_history_tool_call_arguments_invalid_date_fails() -> None:
    runtime = load(VALID / "history_tool_continuation.json")
    args = json.loads(
        runtime["history"][0]["tool_calls"][0]["function"]["arguments"]
    )
    args["service_date"] = "2026-02-31"
    runtime["history"][0]["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
    issues = validate_runtime_input(runtime)
    assert any("invalid tool_call arguments" in i.message for i in issues)
