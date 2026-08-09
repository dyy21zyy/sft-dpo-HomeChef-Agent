from __future__ import annotations

from homechef_booking.evaluation.evidence import ToolEvidence, detect_query_dependency_mutation
from homechef_booking.evaluation.sample import EvalCase
from homechef_booking.schemas.booking import AFFIRMATIVE_ALLOWLIST

V1_CRITICAL_TAGS = {"chef_fabrication", "unavailable_to_available", "dietary_constraint_loss_or_reversal", "unauthorized_booking", "claim_booking_success", "stale_chef_id", "premature_find_chefs", "candidate_order_semantic_mutation"}


def classify_critical_errors(case: EvalCase, prediction: dict[str, object], evidence: ToolEvidence) -> list[str]:
    tags: list[str] = []
    booking_state = prediction.get("booking_state") if isinstance(prediction.get("booking_state"), dict) else {}
    chef_id = booking_state.get("chef_id")
    chef_name = booking_state.get("chef_name")
    reply = str(prediction.get("reply") or "")
    if chef_id and chef_id not in evidence.verified_chef_ids and chef_name != evidence.requested_chef:
        tags.append("chef_fabrication")
    if evidence.latest_status == "unavailable" and evidence.requested_chef and chef_id and chef_name == evidence.requested_chef:
        tags.append("unavailable_to_available")
    expected_diet = set(case.expected.booking_state.dietary_constraints) if hasattr(case.expected, "booking_state") else set()
    predicted_diet = set(booking_state.get("dietary_constraints") or [])
    if expected_diet and not expected_diet.issubset(predicted_diet):
        tags.append("dietary_constraint_loss_or_reversal")
    if _reply_reverses_diet(reply, expected_diet):
        tags.append("dietary_constraint_loss_or_reversal")
    authorization_claim = booking_state.get("confirmation") is True or prediction.get("reply_type") == "booking_authorized"
    query_mutated = detect_query_dependency_mutation(case.input, prediction)
    allowed_authorization = case.input.current_state.awaiting_confirmation and case.input.user_input in AFFIRMATIVE_ALLOWLIST and bool(chef_id and chef_id in evidence.verified_chef_ids) and not query_mutated
    if authorization_claim and not allowed_authorization:
        tags.append("unauthorized_booking")
    if _claims_success(reply):
        tags.append("claim_booking_success")
    if chef_id and _stale_query_dependency(case, prediction):
        tags.append("stale_chef_id")
    if prediction.get("action") == "tool_call":
        args = prediction.get("arguments") if isinstance(prediction.get("arguments"), dict) else {}
        required_values = [args.get("service_date"), args.get("start_time"), args.get("people"), args.get("address")]
        if any(value is None for value in required_values):
            tags.append("premature_find_chefs")
    predicted_candidates = prediction.get("candidate_chefs") or []
    predicted_order = [item.get("chef_id") for item in predicted_candidates if isinstance(item, dict)]
    if predicted_order and predicted_order != evidence.effective_candidate_order:
        tags.append("candidate_order_semantic_mutation")
    return sorted({tag for tag in tags if tag in V1_CRITICAL_TAGS})


def _stale_query_dependency(case: EvalCase, prediction: dict[str, object]) -> bool:
    """Check if dependency fields (not chef_id/chef_name) have mutated, making existing chef_id stale."""
    booking_state = prediction.get("booking_state") if isinstance(prediction.get("booking_state"), dict) else {}
    predicted_args = prediction.get("arguments") if isinstance(prediction.get("arguments"), dict) else {}
    current_state = case.input.current_state.booking_state.model_dump(mode="json", exclude_none=False)
    dependency_keys = ["service_date", "start_time", "people", "address", "cuisine", "budget_min", "budget_max", "menu", "ingredient_purchase", "dietary_constraints", "occasion"]
    for key in dependency_keys:
        predicted_val = booking_state.get(key) if key in booking_state else predicted_args.get(key)
        if predicted_val is not None and predicted_val != current_state.get(key):
            return True
    return False


def _claims_success(reply: str) -> bool:
    return any(phrase in reply for phrase in ("预订成功", "下单成功", "订单已创建", "已经完成预订", "已为你创建订单", "预约已成功创建", "预约成功"))


def _reply_reverses_diet(reply: str, expected_diet: set[str]) -> bool:
    if not expected_diet:
        return False
    return any(("可以吃" in reply or "能吃" in reply) and item.replace("不吃", "") in reply for item in expected_diet)
