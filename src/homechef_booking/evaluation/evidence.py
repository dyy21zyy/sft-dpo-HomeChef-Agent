from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from homechef_booking.schemas.history import AssistantToolCallMessage, ToolMessage
from homechef_booking.schemas.runtime import BookingRuntimeInput
from homechef_booking.schemas.tools import parse_find_chefs_result


class ToolEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    latest_status: str | None = None
    requested_chef: str | None = None
    verified_chef_ids: set[str] = Field(default_factory=set)
    verified_chef_names: set[str] = Field(default_factory=set)
    effective_candidate_order: list[str] = Field(default_factory=list)


def derive_tool_evidence(runtime_input: BookingRuntimeInput) -> ToolEvidence:
    evidence = ToolEvidence()
    pending_args: dict[str, str] = {}
    latest_args: dict[str, object] | None = None
    has_valid_tool_result = False
    for message in runtime_input.history:
        if isinstance(message, AssistantToolCallMessage):
            call = message.tool_calls[0]
            pending_args[call.id] = call.function.arguments
        if isinstance(message, ToolMessage) and message.tool_call_id in pending_args:
            latest_args = json.loads(pending_args[message.tool_call_id])
            result = parse_find_chefs_result(json.loads(message.content)).model_dump(mode="json", exclude_none=False)
            evidence.latest_status = str(result["status"])
            evidence.requested_chef = latest_args.get("chef_name")
            candidates = _candidates_from_result(result)
            evidence.verified_chef_ids = {candidate["chef_id"] for candidate in candidates}
            evidence.verified_chef_names = {candidate["chef_name"] for candidate in candidates}
            evidence.effective_candidate_order = [candidate["chef_id"] for candidate in candidates]
            has_valid_tool_result = True
    state = runtime_input.current_state
    if state.candidate_chefs:
        for chef in state.candidate_chefs:
            evidence.verified_chef_ids.add(chef.chef_id)
            evidence.verified_chef_names.add(chef.chef_name)
            if not has_valid_tool_result and chef.chef_id not in evidence.effective_candidate_order:
                evidence.effective_candidate_order.append(chef.chef_id)
        if not has_valid_tool_result:
            evidence.latest_status = state.chef_query_status.value
    state_chef = state.booking_state
    if state_chef.chef_id and state_chef.chef_name:
        evidence.verified_chef_ids.add(state_chef.chef_id)
        evidence.verified_chef_names.add(state_chef.chef_name)
        if not has_valid_tool_result and state_chef.chef_id not in evidence.effective_candidate_order:
            evidence.effective_candidate_order.append(state_chef.chef_id)
    return evidence


def detect_query_dependency_mutation(runtime_input: BookingRuntimeInput, prediction: dict[str, object]) -> bool:
    predicted_state = prediction.get("booking_state") if isinstance(prediction.get("booking_state"), dict) else {}
    if prediction.get("action") == "tool_call" and isinstance(prediction.get("arguments"), dict):
        predicted_state = prediction["arguments"]
    current_state = runtime_input.current_state.booking_state.model_dump(mode="json", exclude_none=False)
    dependency_keys = ["chef_name", "service_date", "start_time", "people", "address", "cuisine", "budget_min", "budget_max", "menu", "ingredient_purchase", "dietary_constraints", "occasion"]
    return any(predicted_state.get(key) != current_state.get(key) for key in dependency_keys if key in predicted_state)


def _candidates_from_result(result: dict[str, object]) -> list[dict[str, str]]:
    if "candidates" in result:
        return list(result["candidates"])
    if "alternatives" in result:
        return list(result["alternatives"])
    if "chef" in result:
        return [result["chef"]]
    return []
