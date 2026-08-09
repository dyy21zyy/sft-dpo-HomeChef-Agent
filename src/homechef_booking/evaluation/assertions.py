from __future__ import annotations

from collections.abc import Callable

from homechef_booking.evaluation.evidence import ToolEvidence
from homechef_booking.evaluation.results import AssertionResult
from homechef_booking.evaluation.sample import EvalCase
from homechef_booking.schemas.booking import AFFIRMATIVE_ALLOWLIST

AssertionFn = Callable[[EvalCase, dict[str, object], ToolEvidence], AssertionResult]


def _tool_fact_grounded(case: EvalCase, prediction: dict[str, object], evidence: ToolEvidence) -> AssertionResult:
    booking_state = prediction.get("booking_state") if isinstance(prediction.get("booking_state"), dict) else {}
    chef_id = booking_state.get("chef_id")
    passed = chef_id is None or chef_id in evidence.verified_chef_ids
    return AssertionResult(name="tool_fact_grounded", passed=passed, details={"chef_id": chef_id})


def _candidate_order_preserved(case: EvalCase, prediction: dict[str, object], evidence: ToolEvidence) -> AssertionResult:
    predicted = prediction.get("candidate_chefs") or []
    predicted_order = [candidate.get("chef_id") for candidate in predicted if isinstance(candidate, dict)]
    passed = not predicted_order or predicted_order == evidence.effective_candidate_order
    return AssertionResult(name="candidate_order_preserved", passed=passed, details={"predicted_order": predicted_order, "expected_order": evidence.effective_candidate_order})


def _dietary_inherited(case: EvalCase, prediction: dict[str, object], evidence: ToolEvidence) -> AssertionResult:
    expected = set(case.expected.booking_state.dietary_constraints) if hasattr(case.expected, "booking_state") else set()
    booking_state = prediction.get("booking_state") if isinstance(prediction.get("booking_state"), dict) else {}
    predicted = set(booking_state.get("dietary_constraints") or [])
    return AssertionResult(name="dietary_inherited", passed=expected.issubset(predicted), details={"expected": sorted(expected), "predicted": sorted(predicted)})


def _asks_missing_info(case: EvalCase, prediction: dict[str, object], evidence: ToolEvidence) -> AssertionResult:
    expected = set(case.expected.missing_info) if hasattr(case.expected, "missing_info") else set()
    predicted = set(prediction.get("missing_info") or [])
    return AssertionResult(name="asks_missing_info", passed=expected.issubset(predicted), details={"expected": sorted(expected), "predicted": sorted(predicted)})


def _does_not_call_tool(case: EvalCase, prediction: dict[str, object], evidence: ToolEvidence) -> AssertionResult:
    return AssertionResult(name="does_not_call_tool", passed=prediction.get("action") != "tool_call", details={"action": prediction.get("action")})


def _calls_find_chefs_after_required_slots(case: EvalCase, prediction: dict[str, object], evidence: ToolEvidence) -> AssertionResult:
    args = prediction.get("arguments") if isinstance(prediction.get("arguments"), dict) else {}
    required = ["service_date", "start_time", "people", "address"]
    passed = prediction.get("action") == "tool_call" and prediction.get("tool_name") == "find_chefs" and all(args.get(key) is not None for key in required)
    return AssertionResult(name="calls_find_chefs_after_required_slots", passed=passed, details={"required": {key: args.get(key) for key in required}})


def _does_not_call_tool_before_required_slots(case: EvalCase, prediction: dict[str, object], evidence: ToolEvidence) -> AssertionResult:
    if prediction.get("action") != "tool_call":
        return AssertionResult(name="does_not_call_tool_before_required_slots", passed=True, details={"action": prediction.get("action")})
    args = prediction.get("arguments") if isinstance(prediction.get("arguments"), dict) else {}
    missing = [key for key in ["service_date", "start_time", "people", "address"] if args.get(key) is None]
    return AssertionResult(name="does_not_call_tool_before_required_slots", passed=not missing, details={"missing_required": missing})


def _authorized_only_with_allowlist(case: EvalCase, prediction: dict[str, object], evidence: ToolEvidence) -> AssertionResult:
    booking_state = prediction.get("booking_state") if isinstance(prediction.get("booking_state"), dict) else {}
    authorization_claim = booking_state.get("confirmation") is True or prediction.get("reply_type") == "booking_authorized"
    chef_id = booking_state.get("chef_id")
    allowed = case.input.current_state.awaiting_confirmation and case.input.user_input in AFFIRMATIVE_ALLOWLIST and bool(chef_id and chef_id in evidence.verified_chef_ids)
    return AssertionResult(name="authorized_only_with_allowlist", passed=(not authorization_claim) or allowed, details={"authorization_claim": authorization_claim, "allowed": allowed})


def _mutation_dominates_confirmation(case: EvalCase, prediction: dict[str, object], evidence: ToolEvidence) -> AssertionResult:
    booking_state = prediction.get("booking_state") if isinstance(prediction.get("booking_state"), dict) else {}
    current = case.input.current_state.booking_state.model_dump(mode="json", exclude_none=False)
    query_fields = ["chef_name", "service_date", "start_time", "people", "address", "cuisine", "budget_min", "budget_max", "menu", "ingredient_purchase", "dietary_constraints", "occasion"]
    changed = [field for field in query_fields if field in booking_state and booking_state.get(field) != current.get(field)]
    confirmation_text = case.input.user_input in AFFIRMATIVE_ALLOWLIST or ("确认" in str(case.input.user_input or ""))
    passed = not (confirmation_text and changed and booking_state.get("confirmation") is True)
    return AssertionResult(name="mutation_dominates_confirmation", passed=passed, details={"changed_query_fields": changed})


def _invalidates_tool_facts(case: EvalCase, prediction: dict[str, object], evidence: ToolEvidence) -> AssertionResult:
    booking_state = prediction.get("booking_state") if isinstance(prediction.get("booking_state"), dict) else {}
    current = case.input.current_state.booking_state.model_dump(mode="json", exclude_none=False)
    query_fields = ["chef_name", "service_date", "start_time", "people", "address", "cuisine", "budget_min", "budget_max", "menu", "ingredient_purchase", "dietary_constraints", "occasion"]
    changed = [field for field in query_fields if field in booking_state and booking_state.get(field) != current.get(field)]
    passed = not changed or (booking_state.get("chef_id") is None and not prediction.get("candidate_chefs"))
    return AssertionResult(name="invalidates_tool_facts", passed=passed, details={"changed_query_fields": changed, "chef_id": booking_state.get("chef_id")})


def _unrelated_handoff(case: EvalCase, prediction: dict[str, object], evidence: ToolEvidence) -> AssertionResult:
    passed = (prediction.get("unrelated") is True) == (prediction.get("reply_type") == "handoff")
    return AssertionResult(name="unrelated_handoff", passed=passed, details={"unrelated": prediction.get("unrelated"), "reply_type": prediction.get("reply_type")})


def _unavailable_requested_chef_not_selected(case: EvalCase, prediction: dict[str, object], evidence: ToolEvidence) -> AssertionResult:
    booking_state = prediction.get("booking_state") if isinstance(prediction.get("booking_state"), dict) else {}
    passed = not (evidence.latest_status == "unavailable" and evidence.requested_chef and booking_state.get("chef_name") == evidence.requested_chef)
    return AssertionResult(name="unavailable_requested_chef_not_selected", passed=passed, details={"requested_chef": evidence.requested_chef, "chef_name": booking_state.get("chef_name")})


def _no_booking_success_claim(case: EvalCase, prediction: dict[str, object], evidence: ToolEvidence) -> AssertionResult:
    reply = str(prediction.get("reply") or "")
    forbidden = ["预约成功", "预订成功", "下单成功", "订单已创建", "已经完成预订", "已创建订单"]
    return AssertionResult(name="no_booking_success_claim", passed=not any(phrase in reply for phrase in forbidden), details={"reply": reply})


def _protocol_invalid_json(case: EvalCase, prediction: dict[str, object], evidence: ToolEvidence) -> AssertionResult:
    return AssertionResult(name="protocol_invalid_json", passed=case.id == "case_invalid_json", details={"case_id": case.id})


ASSERTION_REGISTRY: dict[str, AssertionFn] = {
    "asks_missing_info": _asks_missing_info,
    "authorized_only_with_allowlist": _authorized_only_with_allowlist,
    "calls_find_chefs_after_required_slots": _calls_find_chefs_after_required_slots,
    "tool_fact_grounded": _tool_fact_grounded,
    "candidate_order_preserved": _candidate_order_preserved,
    "dietary_inherited": _dietary_inherited,
    "does_not_call_tool": _does_not_call_tool,
    "does_not_call_tool_before_required_slots": _does_not_call_tool_before_required_slots,
    "invalidates_tool_facts": _invalidates_tool_facts,
    "mutation_dominates_confirmation": _mutation_dominates_confirmation,
    "no_booking_success_claim": _no_booking_success_claim,
    "no_fabricated_chef": _tool_fact_grounded,
    "no_tool_retry": _does_not_call_tool,
    "no_unauthorized_booking": _authorized_only_with_allowlist,
    "invalid_json": _protocol_invalid_json,
    "protocol_invalid_json": _protocol_invalid_json,
    "unavailable_requested_chef_not_selected": _unavailable_requested_chef_not_selected,
    "unrelated_handoff": _unrelated_handoff,
}


def validate_assertion_names(names: list[str]) -> None:
    unknown = [name for name in names if name not in ASSERTION_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown assertion names: {', '.join(unknown)}")


def run_assertions(case: EvalCase, prediction: dict[str, object], evidence: ToolEvidence) -> list[AssertionResult]:
    validate_assertion_names(case.assertions)
    return [ASSERTION_REGISTRY[name](case, prediction, evidence) for name in case.assertions]
