"""Phase 03 v0.2 — Scenario Lineage / Business Fact single source of truth.

This module establishes a deterministic typed lineage from a canonical business
state through tool arguments, tool results, and expected final decisions.

Lineage chain:

    Initial / Current Booking State
            ↓
    User Request / Modification
            ↓
    Effective Query State
            ↓
    Assistant Tool Call (arguments derived from query state)
            ↓
    Tool Result
            ↓
    Expected Final Decision (booking_state, chef_query_status, candidates, etc.)

All fields in a single Raw sample must trace back to the same Effective Query
State.  No per-field patching of ``expected`` is allowed — the entire sample
is derived from :class:`ScenarioFacts`.

The typed models here reuse the project's real contract types
(:class:`BookingSlot`, :class:`DecisionState`, :class:`CandidateChef`,
:class:`FindChefsInput`, Tool Result models) rather than creating a second
business schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from homechef_booking.schemas.booking import (
    BookingSlot,
    CandidateChef,
    ChefQueryStatus,
    ReplyType,
    missing_required_slots,
)

# ── Tool Result status → chef_query_status deterministic mapping ────────────

# The canonical mapping from a find_chefs Tool Result (mode, status) pair to
# the ChefQueryStatus that the FinalDecision must carry.
#
# This is the ONLY place where this mapping is defined.  Both the generator
# and the validators must import it.
TOOL_RESULT_STATUS_TO_QUERY_STATUS: dict[tuple[str, str], ChefQueryStatus] = {
    ("search", "matched"): ChefQueryStatus.matched,
    ("search", "no_match"): ChefQueryStatus.no_match,
    ("search", "out_of_service_area"): ChefQueryStatus.out_of_service_area,
    ("search", "error"): ChefQueryStatus.error,
    ("specific", "available"): ChefQueryStatus.available,
    ("specific", "unavailable"): ChefQueryStatus.unavailable,
    ("specific", "not_found"): ChefQueryStatus.not_found,
    ("specific", "out_of_service_area"): ChefQueryStatus.out_of_service_area,
    ("specific", "error"): ChefQueryStatus.error,
}


# ── Tool Result status → reply_type deterministic mapping ───────────────────

TOOL_RESULT_STATUS_TO_REPLY_TYPE: dict[tuple[str, str], ReplyType] = {
    ("search", "matched"): ReplyType.present_chef_candidates,
    ("search", "no_match"): ReplyType.inform_no_match,
    ("search", "out_of_service_area"): ReplyType.inform_out_of_service_area,
    ("search", "error"): ReplyType.booking_paused,
    ("specific", "available"): ReplyType.confirm_specific_chef,
    ("specific", "unavailable"): ReplyType.present_alternatives,
    ("specific", "not_found"): ReplyType.inform_not_found,
    ("specific", "out_of_service_area"): ReplyType.inform_out_of_service_area,
    ("specific", "error"): ReplyType.booking_paused,
}


# ── ScenarioFacts: the single source of truth ───────────────────────────────


@dataclass
class ScenarioFacts:
    """Deterministic business-fact lineage for a single Raw sample.

    Every field that ends up in ``input.current_state``, ``input.history``,
    and ``expected`` is derived from this object.  Nothing is hand-written
    per-field in the generator.

    Fields:
        booking_state: The effective booking state AFTER applying the user
            request (i.e. the state the tool call arguments must match).
        tool_mode: ``"search"`` / ``"specific"`` / ``""`` (no tool).
        tool_result_status: status string from the Tool Result contract,
            or ``""`` when no tool result exists.
        tool_result_payload: the full Tool Result dict (mode, status, plus
            candidates/chef/alternatives/error fields).  ``None`` when no
            tool result exists.
        selected_chef: the :class:`CandidateChef` the user selected (if any).
        requested_chef_name: for specific-mode queries, the chef name.
        user_confirms: whether the user gave an affirmative confirmation.
        scenario: scenario label for tags / semantic validators.
    """

    booking_state: BookingSlot
    tool_mode: str = ""
    tool_result_status: str = ""
    tool_result_payload: dict[str, Any] | None = None
    selected_chef: CandidateChef | None = None
    requested_chef_name: str | None = None
    user_confirms: bool = False
    user_rejects: bool = False
    scenario: str = ""

    # ── Derived properties (deterministic, no hand-writing) ──────────────

    @property
    def effective_query_state(self) -> BookingSlot:
        """The booking_state whose slots drive the tool call arguments."""
        return self.booking_state

    @property
    def has_tool_call(self) -> bool:
        return self.tool_mode != ""

    @property
    def has_tool_result(self) -> bool:
        return self.tool_result_payload is not None

    @property
    def effective_query_status(self) -> ChefQueryStatus:
        """Derive chef_query_status from tool result, or not_checked."""
        if not self.has_tool_result:
            return ChefQueryStatus.not_checked
        key = (self.tool_mode, self.tool_result_status)
        return TOOL_RESULT_STATUS_TO_QUERY_STATUS.get(key, ChefQueryStatus.not_checked)

    @property
    def effective_candidates(self) -> list[CandidateChef]:
        """Derive candidate_chefs from the tool result payload."""
        if not self.has_tool_result:
            return []
        payload = self.tool_result_payload
        status = self.tool_result_status
        if status == "matched":
            raw = payload.get("candidates", [])
            return [CandidateChef.model_validate(c) for c in raw]
        if status == "unavailable":
            raw = payload.get("alternatives", [])
            return [CandidateChef.model_validate(c) for c in raw]
        if status == "available":
            chef = payload.get("chef")
            if chef:
                return [CandidateChef.model_validate(chef)]
            return []
        return []

    @property
    def effective_chef_id(self) -> str | None:
        """Derive the booking_state.chef_id from evidence."""
        if self.selected_chef is not None:
            return self.selected_chef.chef_id
        if self.tool_result_status == "available":
            chef = (self.tool_result_payload or {}).get("chef")
            if chef:
                return chef.get("chef_id")
        return None

    @property
    def effective_chef_name(self) -> str | None:
        """Derive the booking_state.chef_name from evidence."""
        if self.selected_chef is not None:
            return self.selected_chef.chef_name
        if self.requested_chef_name is not None:
            return self.requested_chef_name
        if self.tool_result_status == "available":
            chef = (self.tool_result_payload or {}).get("chef")
            if chef:
                return chef.get("chef_name")
        return None

    @property
    def effective_missing_info(self) -> list[str]:
        """The canonical missing required slots from booking_state."""
        return missing_required_slots(self.booking_state)

    @property
    def effective_info_complete(self) -> bool:
        """Derive info_complete from missing_required_slots()."""
        return len(self.effective_missing_info) == 0

    @property
    def effective_reply_type(self) -> ReplyType:
        """Derive reply_type from tool result or booking completeness."""
        if self.user_confirms and self.tool_result_status == "available":
            return ReplyType.booking_authorized
        if self.user_rejects:
            return ReplyType.booking_paused
        if self.has_tool_result:
            key = (self.tool_mode, self.tool_result_status)
            rt = TOOL_RESULT_STATUS_TO_REPLY_TYPE.get(key)
            if rt is not None:
                return rt
        # No tool result: ask for missing slots or handoff
        if self.scenario == "unrelated":
            return ReplyType.handoff
        missing = self.effective_missing_info
        if len(missing) == 0:
            return ReplyType.acknowledge_result
        if len(missing) == 1:
            slot = missing[0]
            return {
                "service_date": ReplyType.ask_service_date,
                "start_time": ReplyType.ask_start_time,
                "people": ReplyType.ask_people,
                "address": ReplyType.ask_address,
            }[slot]
        return ReplyType.ask_multiple_required_fields

    @property
    def effective_unrelated(self) -> bool:
        return self.scenario == "unrelated"

    @property
    def effective_awaiting_confirmation(self) -> bool:
        """Whether current_state.awaiting_confirmation is True.

        awaiting_confirmation reflects whether the agent was waiting for the
        user to confirm before this turn's user_input.  It is True when:
        - a specific chef was found available (tool result specific/available)
          and the user has NOT yet confirmed in this turn, OR
        - the user is about to confirm (current_state is pre-confirmation).
        """
        if self.user_confirms:
            # User confirms this turn → current_state was awaiting_confirmation
            return True
        if self.tool_result_status == "available":
            # Chef available, user hasn't confirmed yet → awaiting
            return True
        return False


# ── Deterministic builders from ScenarioFacts ───────────────────────────────


def build_find_chefs_arguments(facts: ScenarioFacts) -> dict[str, Any]:
    """Build the 12-key find_chefs arguments dict from the query state.

    This is the deterministic bridge: booking_state → tool_call arguments.
    The generator must never write tool arguments independently.
    """
    bs = facts.effective_query_state
    return {
        "chef_name": bs.chef_name,
        "service_date": bs.service_date,
        "start_time": bs.start_time,
        "people": bs.people,
        "address": bs.address,
        "cuisine": bs.cuisine,
        "budget_min": bs.budget_min,
        "budget_max": bs.budget_max,
        "menu": list(bs.menu),
        "ingredient_purchase": bs.ingredient_purchase,
        "dietary_constraints": list(bs.dietary_constraints),
        "occasion": bs.occasion,
    }


def build_history_with_tool_result(
    user_msg: str,
    facts: ScenarioFacts,
    call_id: str = "call_001",
) -> list[dict[str, Any]]:
    """Build a typed history sequence: user → assistant tool_call → tool result.

    The tool_call arguments are derived from ``facts.effective_query_state``
    via :func:`build_find_chefs_arguments`.  The tool result content is
    ``facts.tool_result_payload`` serialized to JSON.
    """
    args = build_find_chefs_arguments(facts)
    tool_result = facts.tool_result_payload
    return [
        {"role": "user", "content": user_msg},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "find_chefs",
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "find_chefs",
            "content": json.dumps(tool_result, ensure_ascii=False),
        },
    ]


def build_current_state(facts: ScenarioFacts) -> dict[str, Any]:
    """Build the current_state dict from ScenarioFacts.

    For tool_result samples, current_state reflects the post-tool-result state:
    booking_state carries the query slots, chef_query_status reflects the tool
    result, and candidate_chefs carry the tool result candidates.
    """
    bs = facts.booking_state.model_dump(mode="json")
    # Apply evidence-derived chef fields
    if facts.effective_chef_id is not None:
        bs["chef_id"] = facts.effective_chef_id
    if facts.effective_chef_name is not None:
        bs["chef_name"] = facts.effective_chef_name
    return {
        "booking_state": bs,
        "chef_query_status": facts.effective_query_status.value,
        "candidate_chefs": [c.model_dump(mode="json") for c in facts.effective_candidates],
        "awaiting_confirmation": facts.effective_awaiting_confirmation,
    }


def build_expected_final(facts: ScenarioFacts) -> dict[str, Any]:
    """Build the expected FinalDecision dict from ScenarioFacts.

    All fields are deterministic:
    - booking_state: query state + evidence-derived chef fields + confirmation
    - chef_query_status: from tool result mapping
    - candidate_chefs: from tool result candidates
    - info_complete: from missing_required_slots()
    - missing_info: from missing_required_slots()
    - reply_type: from tool result mapping or booking completeness
    """
    bs = facts.booking_state.model_dump(mode="json")
    if facts.effective_chef_id is not None:
        bs["chef_id"] = facts.effective_chef_id
    if facts.effective_chef_name is not None:
        bs["chef_name"] = facts.effective_chef_name
    bs["confirmation"] = (
        True if facts.user_confirms
        else (False if (facts.tool_result_status == "error" or facts.user_rejects) else None)
    )

    return {
        "action": "final",
        "booking_state": bs,
        "chef_query_status": facts.effective_query_status.value,
        "candidate_chefs": [c.model_dump(mode="json") for c in facts.effective_candidates],
        "info_complete": facts.effective_info_complete,
        "unrelated": facts.effective_unrelated,
        "missing_info": list(facts.effective_missing_info),
        "reply_type": facts.effective_reply_type.value,
        "reply": "",
    }


def build_expected_tool_call(facts: ScenarioFacts) -> dict[str, Any]:
    """Build the expected ToolCallDecision dict from ScenarioFacts."""
    return {
        "action": "tool_call",
        "tool_name": "find_chefs",
        "arguments": build_find_chefs_arguments(facts),
    }


def derive_query_state_from_tool_result(
    tool_result: dict[str, Any],
    base_booking: BookingSlot,
) -> BookingSlot:
    """Derive the effective query state from a tool result + base booking.

    For specific-mode queries, the chef_name from the tool result's
    requested_chef or chef field is set on the booking state.  For search-mode
    queries, the booking state is the query state as-is.

    This helper exists so validators can independently reconstruct the query
    state from a tool result and compare it to the sample's booking_state.
    """
    bs = base_booking.model_copy(deep=True)
    mode = tool_result.get("mode")
    if mode == "specific":
        requested = tool_result.get("requested_chef")
        if requested:
            bs.chef_name = requested
        elif tool_result.get("status") == "available":
            chef = tool_result.get("chef", {})
            if isinstance(chef, dict) and chef.get("chef_name"):
                bs.chef_name = chef["chef_name"]
    return bs


__all__ = [
    "ScenarioFacts",
    "TOOL_RESULT_STATUS_TO_QUERY_STATUS",
    "TOOL_RESULT_STATUS_TO_REPLY_TYPE",
    "build_current_state",
    "build_expected_final",
    "build_expected_tool_call",
    "build_find_chefs_arguments",
    "build_history_with_tool_result",
    "derive_query_state_from_tool_result",
]
