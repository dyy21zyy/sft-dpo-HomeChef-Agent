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
    FinalDecision,
    ToolCallDecision,
    parse_decision_obj,
)
from homechef_booking.schemas.runtime import _FIND_CHEFS_REQUIRED_KEYS  # noqa: F401
from homechef_booking.schemas.tools import FindChefsInput, parse_find_chefs_result
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
    decision = load(VALID / "tool_call_decision.json")
    decision["arguments"]["service_date"] = None
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
        "chef_query_status": "matched",
        "candidate_chefs": [
            {"chef_id": "C003", "chef_name": "张伟"},
        ],
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


# --- Fix 3: booking_authorized with status=matched / status=unavailable ---


def test_booking_authorized_matched_candidate_status_matched() -> None:
    """Matched candidate → confirmation → booking_authorized with status=matched PASS."""
    runtime = load(VALID / "history_tool_continuation.json")
    runtime["current_state"]["awaiting_confirmation"] = True
    runtime["current_state"]["booking_state"]["chef_id"] = "C003"
    runtime["current_state"]["booking_state"]["chef_name"] = "张伟"
    runtime["user_input"] = "确认"
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
        "chef_query_status": "matched",
        "candidate_chefs": [
            {"chef_id": "C003", "chef_name": "张伟"},
            {"chef_id": "C007", "chef_name": "李明"},
        ],
        "info_complete": True,
        "unrelated": False,
        "missing_info": [],
        "reply_type": "booking_authorized",
        "reply": "好的，已为您预约张伟厨师。",
    }
    issues = validate_decision(decision, runtime)
    assert issues == []


def test_booking_authorized_unavailable_alternative_status_unavailable() -> None:
    """Unavailable alternative → confirmation → booking_authorized with status=unavailable PASS."""
    runtime = load(VALID / "history_tool_continuation.json")
    runtime["current_state"]["awaiting_confirmation"] = True
    runtime["current_state"]["chef_query_status"] = "unavailable"
    runtime["current_state"]["candidate_chefs"] = [
        {"chef_id": "C005", "chef_name": "王芳"},
    ]
    runtime["current_state"]["booking_state"]["chef_id"] = "C005"
    runtime["current_state"]["booking_state"]["chef_name"] = "王芳"
    runtime["user_input"] = "确认"
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
            "chef_id": "C005",
            "chef_name": "王芳",
            "ingredient_purchase": None,
            "dietary_constraints": [],
            "occasion": "家庭聚餐",
            "confirmation": True,
        },
        "chef_query_status": "unavailable",
        "candidate_chefs": [
            {"chef_id": "C005", "chef_name": "王芳"},
        ],
        "info_complete": True,
        "unrelated": False,
        "missing_info": [],
        "reply_type": "booking_authorized",
        "reply": "好的，已为您预约王芳厨师。",
    }
    issues = validate_decision(decision, runtime)
    assert issues == []


# --- Fix 5: ToolCall checks decision.arguments, not current_state ---


def test_tool_call_checks_arguments_not_current_state() -> None:
    """current_state missing address, but ToolCall.arguments has address → PASS."""
    runtime = load(VALID / "minimal_runtime_input.json")
    runtime["current_state"]["booking_state"]["address"] = None
    decision = load(VALID / "tool_call_decision.json")
    issues = validate_decision(decision, runtime)
    assert not any("required slots" in i.message for i in issues)


# --- Fix 7: info_complete=1 (non-boolean) → FAIL ---


def test_info_complete_non_bool_fails() -> None:
    with pytest.raises(ValidationError):
        FinalDecision.model_validate({
            "action": "final",
            "booking_state": {
                "service_date": None,
                "start_time": None,
                "people": None,
                "address": None,
                "cuisine": None,
                "budget_min": None,
                "budget_max": None,
                "menu": [],
                "chef_id": None,
                "chef_name": None,
                "ingredient_purchase": None,
                "dietary_constraints": [],
                "occasion": None,
                "confirmation": None,
            },
            "chef_query_status": "not_checked",
            "candidate_chefs": [],
            "info_complete": 1,
            "unrelated": False,
            "missing_info": ["service_date"],
            "reply_type": "ask_service_date",
            "reply": "ok",
        })


# --- Fix 7: invalid action → clean FAIL, no crash ---


def test_invalid_action_clean_failure() -> None:
    with pytest.raises(ValidationError):
        parse_decision_obj({"action": "invalid_action"})


# --- CG-04 new tests ---


def test_matched_empty_candidates_fails() -> None:
    with pytest.raises(ValidationError):
        parse_find_chefs_result({"mode": "search", "status": "matched", "candidates": []})


def test_no_match_missing_candidates_fails() -> None:
    with pytest.raises(ValidationError):
        parse_find_chefs_result({"mode": "search", "status": "no_match"})


def test_no_match_nonempty_candidates_fails() -> None:
    with pytest.raises(ValidationError):
        parse_find_chefs_result({
            "mode": "search",
            "status": "no_match",
            "candidates": [{"chef_id": "C003", "chef_name": "张伟"}],
        })


def test_specific_available_missing_chef_fails() -> None:
    with pytest.raises(ValidationError):
        parse_find_chefs_result({"mode": "specific", "status": "available"})


def test_specific_unavailable_missing_alternatives_fails() -> None:
    with pytest.raises(ValidationError):
        parse_find_chefs_result({
            "mode": "specific",
            "status": "unavailable",
            "requested_chef": "张伟",
        })


def test_specific_unavailable_empty_alternatives_passes() -> None:
    result = parse_find_chefs_result({
        "mode": "specific",
        "status": "unavailable",
        "requested_chef": "张伟",
        "alternatives": [],
    })
    assert result.status == "unavailable"


def test_error_missing_required_fields_fails() -> None:
    with pytest.raises(ValidationError):
        parse_find_chefs_result({"mode": "search", "status": "error"})


def test_candidate_chef_extra_field_fails() -> None:
    with pytest.raises(ValidationError):
        parse_find_chefs_result({
            "mode": "search",
            "status": "matched",
            "candidates": [
                {"chef_id": "C003", "chef_name": "张伟", "extra": "forbidden"},
            ],
        })


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


# --- Fix 1: empty alternatives constraints ---


def test_not_found_nonempty_alternatives_fails() -> None:
    with pytest.raises(ValidationError):
        parse_find_chefs_result({
            "mode": "specific",
            "status": "not_found",
            "requested_chef": "张伟",
            "alternatives": [{"chef_id": "C005", "chef_name": "王芳"}],
        })


def test_out_of_service_area_nonempty_alternatives_fails() -> None:
    with pytest.raises(ValidationError):
        parse_find_chefs_result({
            "mode": "specific",
            "status": "out_of_service_area",
            "requested_chef": "张伟",
            "alternatives": [{"chef_id": "C005", "chef_name": "王芳"}],
        })


# --- Fix 2: full available_tools contract ---


def test_available_tools_empty_parameters_fails() -> None:
    runtime = load(VALID / "minimal_runtime_input.json")
    runtime["available_tools"][0]["function"]["parameters"] = {}
    issues = validate_runtime_input(runtime)
    assert any("parameters" in i.message for i in issues)


def test_available_tools_missing_required_key_fails() -> None:
    runtime = load(VALID / "minimal_runtime_input.json")
    params = runtime["available_tools"][0]["function"]["parameters"]
    params["required"] = ["chef_name"]
    issues = validate_runtime_input(runtime)
    assert any("required" in i.message for i in issues)


def test_available_tools_extra_property_key_fails() -> None:
    runtime = load(VALID / "minimal_runtime_input.json")
    params = runtime["available_tools"][0]["function"]["parameters"]
    params["properties"]["city"] = {"type": "string"}
    issues = validate_runtime_input(runtime)
    assert any("properties" in i.message for i in issues)


def test_available_tools_wrong_function_name_fails() -> None:
    runtime = load(VALID / "minimal_runtime_input.json")
    runtime["available_tools"][0]["function"]["name"] = "wrong_tool"
    issues = validate_runtime_input(runtime)
    assert any("name" in i.path for i in issues)


# --- Fix 3: specific/available chef provenance ---


def test_specific_available_chef_c003_decision_c003_passes() -> None:
    runtime = load(VALID / "history_tool_continuation.json")
    runtime["current_state"]["awaiting_confirmation"] = True
    runtime["current_state"]["booking_state"]["chef_id"] = "C003"
    runtime["current_state"]["booking_state"]["chef_name"] = "张伟"
    runtime["current_state"]["chef_query_status"] = "available"
    runtime["current_state"]["candidate_chefs"] = []
    runtime["user_input"] = "确认"
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
        "reply": "好的，已为您预约张伟厨师。",
    }
    issues = validate_decision(decision, runtime)
    assert issues == []


def test_specific_available_chef_c003_decision_c999_fails() -> None:
    runtime = load(VALID / "history_tool_continuation.json")
    runtime["current_state"]["awaiting_confirmation"] = True
    runtime["current_state"]["booking_state"]["chef_id"] = "C003"
    runtime["current_state"]["booking_state"]["chef_name"] = "张伟"
    runtime["current_state"]["chef_query_status"] = "available"
    runtime["user_input"] = "确认"
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
            "chef_id": "C999",
            "chef_name": "虚构厨师",
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
    assert any("verified chef" in i.message for i in issues)


def test_no_provenance_arbitrary_chef_id_fails() -> None:
    runtime = load(VALID / "minimal_runtime_input.json")
    runtime["current_state"]["awaiting_confirmation"] = True
    runtime["user_input"] = "确认"
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
            "chef_id": "C999",
            "chef_name": "虚构厨师",
            "ingredient_purchase": None,
            "dietary_constraints": [],
            "occasion": "家庭聚餐",
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
    assert any("verified chef" in i.message for i in issues)


# --- Fix 4: budget JSON number semantics ---


def test_budget_accepts_int_and_float() -> None:
    BookingSlot(budget_min=800, budget_max=1200.0)
    FindChefsInput.model_validate({
        "chef_name": None,
        "service_date": "2026-08-15",
        "start_time": "18:00",
        "people": 6,
        "address": "杨浦",
        "cuisine": "川菜",
        "budget_min": 800,
        "budget_max": 1200.0,
        "menu": [],
        "ingredient_purchase": None,
        "dietary_constraints": [],
        "occasion": "家庭聚餐",
    })


def test_budget_rejects_string_and_bool() -> None:
    with pytest.raises(ValidationError):
        BookingSlot(budget_min="800")
    with pytest.raises(ValidationError):
        BookingSlot(budget_max=True)


# --- ToolSpec parity regression tests ---


def _make_runtime_with_params(params: dict) -> dict:
    runtime = load(VALID / "minimal_runtime_input.json")
    runtime["available_tools"][0]["function"]["parameters"] = params
    return runtime


def test_toolspec_type_other_fails() -> None:
    runtime = load(VALID / "minimal_runtime_input.json")
    runtime["available_tools"][0]["type"] = "other"
    issues = validate_runtime_input(runtime)
    assert any("type" in i.path for i in issues)


def test_toolspec_function_name_other_fails() -> None:
    runtime = load(VALID / "minimal_runtime_input.json")
    runtime["available_tools"][0]["function"]["name"] = "other"
    issues = validate_runtime_input(runtime)
    assert any("name" in i.path for i in issues)


def test_toolspec_people_type_string_fails() -> None:
    params = {
        "type": "object",
        "additionalProperties": False,
        "required": list(sorted(_FIND_CHEFS_REQUIRED_KEYS)),  # noqa: F821
        "properties": {
            "chef_name": {"type": ["string", "null"]},
            "service_date": {
                "type": ["string", "null"],
                "pattern": r"^\d{4}-\d{2}-\d{2}$",
                "format": "date",
            },
            "start_time": {
                "type": ["string", "null"],
                "pattern": r"^([01]\d|2[0-3]):[0-5]\d$",
            },
            "people": {"type": ["string", "null"]},
            "address": {"type": ["string", "null"]},
            "cuisine": {"type": ["string", "null"]},
            "budget_min": {"type": ["number", "null"]},
            "budget_max": {"type": ["number", "null"]},
            "menu": {"type": "array", "items": {"type": "string"}},
            "ingredient_purchase": {"type": ["boolean", "null"]},
            "dietary_constraints": {"type": "array", "items": {"type": "string"}},
            "occasion": {"type": ["string", "null"]},
        },
    }
    runtime = _make_runtime_with_params(params)
    issues = validate_runtime_input(runtime)
    assert any("people" in i.message for i in issues)


def test_toolspec_service_date_missing_format_fails() -> None:
    params = {
        "type": "object",
        "additionalProperties": False,
        "required": list(sorted(_FIND_CHEFS_REQUIRED_KEYS)),  # noqa: F821
        "properties": {
            "chef_name": {"type": ["string", "null"]},
            "service_date": {"type": ["string", "null"]},
            "start_time": {
                "type": ["string", "null"],
                "pattern": r"^([01]\d|2[0-3]):[0-5]\d$",
            },
            "people": {"type": ["integer", "null"]},
            "address": {"type": ["string", "null"]},
            "cuisine": {"type": ["string", "null"]},
            "budget_min": {"type": ["number", "null"]},
            "budget_max": {"type": ["number", "null"]},
            "menu": {"type": "array", "items": {"type": "string"}},
            "ingredient_purchase": {"type": ["boolean", "null"]},
            "dietary_constraints": {"type": "array", "items": {"type": "string"}},
            "occasion": {"type": ["string", "null"]},
        },
    }
    runtime = _make_runtime_with_params(params)
    issues = validate_runtime_input(runtime)
    assert any("service_date" in i.message for i in issues)


def test_toolspec_start_time_wrong_pattern_fails() -> None:
    params = {
        "type": "object",
        "additionalProperties": False,
        "required": list(sorted(_FIND_CHEFS_REQUIRED_KEYS)),  # noqa: F821
        "properties": {
            "chef_name": {"type": ["string", "null"]},
            "service_date": {
                "type": ["string", "null"],
                "pattern": r"^\d{4}-\d{2}-\d{2}$",
                "format": "date",
            },
            "start_time": {"type": ["string", "null"], "pattern": ".*"},
            "people": {"type": ["integer", "null"]},
            "address": {"type": ["string", "null"]},
            "cuisine": {"type": ["string", "null"]},
            "budget_min": {"type": ["number", "null"]},
            "budget_max": {"type": ["number", "null"]},
            "menu": {"type": "array", "items": {"type": "string"}},
            "ingredient_purchase": {"type": ["boolean", "null"]},
            "dietary_constraints": {"type": "array", "items": {"type": "string"}},
            "occasion": {"type": ["string", "null"]},
        },
    }
    runtime = _make_runtime_with_params(params)
    issues = validate_runtime_input(runtime)
    assert any("start_time" in i.message for i in issues)


def test_toolspec_correct_complete_schema_passes() -> None:
    params = {
        "type": "object",
        "additionalProperties": False,
        "required": list(sorted(_FIND_CHEFS_REQUIRED_KEYS)),  # noqa: F821
        "properties": {
            "chef_name": {"type": ["string", "null"]},
            "service_date": {
                "type": ["string", "null"],
                "pattern": r"^\d{4}-\d{2}-\d{2}$",
                "format": "date",
            },
            "start_time": {
                "type": ["string", "null"],
                "pattern": r"^([01]\d|2[0-3]):[0-5]\d$",
            },
            "people": {"type": ["integer", "null"]},
            "address": {"type": ["string", "null"]},
            "cuisine": {"type": ["string", "null"]},
            "budget_min": {"type": ["number", "null"]},
            "budget_max": {"type": ["number", "null"]},
            "menu": {"type": "array", "items": {"type": "string"}},
            "ingredient_purchase": {"type": ["boolean", "null"]},
            "dietary_constraints": {"type": "array", "items": {"type": "string"}},
            "occasion": {"type": ["string", "null"]},
        },
    }
    runtime = _make_runtime_with_params(params)
    issues = validate_runtime_input(runtime)
    assert issues == []
