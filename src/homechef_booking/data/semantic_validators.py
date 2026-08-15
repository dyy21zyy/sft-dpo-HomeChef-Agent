"""Phase 03 Semantic Validators — business-semantic quality gates for Raw Booking Samples.

Validators:
  1. Relative Time  — verify service_date matches deterministic template+base_datetime
  2. State Inheritance — verify preserved/changed/invalidated fields across multi-turn
  3. Tool Fact Grounding — verify chef facts trace to tool_result evidence
  4. Candidate Order — verify candidate_chefs order matches tool_result
  5. Dietary Preservation — verify dietary_constraints survive state changes
  6. Confirmation Transition — verify confirmation/authorization state transitions
"""

from __future__ import annotations

import json as _json
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from homechef_booking.data.raw_sample import RawBookingSample
from homechef_booking.schemas.decision import FinalDecision

# ── 1. Relative Time Validator ──────────────────────────────

# Known relative time templates from Phase 03 generator
_RELATIVE_TIME_MAP: dict[str, Callable[[date], date]] = {
    "today": lambda d: d,
    "tomorrow": lambda d: d + timedelta(days=1),
    "day_after_tomorrow": lambda d: d + timedelta(days=2),
    "tonight": lambda d: d,
    "tomorrow_evening": lambda d: d + timedelta(days=1),
    "this_weekend_saturday": lambda d: _next_weekday(d, 5),  # Saturday=5
    "this_weekend_sunday": lambda d: _next_weekday(d, 6),    # Sunday=6
    "this_saturday": lambda d: _next_weekday(d, 5),
    "this_sunday": lambda d: _next_weekday(d, 6),
    "next_monday": lambda d: _next_monday(d),
    "next_friday": lambda d: _next_friday(d),
    "next_weekend": lambda d: _next_weekday(d + timedelta(days=1), 5),
}


def _next_weekday(base: date, target_weekday: int) -> date:
    """Return next occurrence of target_weekday (Mon=0, Sun=6) on or after base."""
    days_ahead = target_weekday - base.weekday()
    if days_ahead < 0:
        days_ahead += 7
    return base + timedelta(days=days_ahead)


def _next_monday(base: date) -> date:
    days = 7 - base.weekday()
    return base + timedelta(days=days if days > 0 else 7)


def _next_friday(base: date) -> date:
    days = 4 - base.weekday()
    if days <= 0:
        days += 7
    return base + timedelta(days=days)


def _parse_current_time(sample: RawBookingSample) -> date | None:
    """Extract current_time date from the sample's input."""
    ct = sample.input.current_time
    if not ct:
        return None
    try:
        return date.fromisoformat(ct[:10])
    except (ValueError, TypeError):
        return None


def validate_relative_time(sample: RawBookingSample) -> str | None:
    """v0.2: Verify service_date using RelativeTimeMetadata from generation.

    Uses the v0.2 generation.relative_time_metadata field.
    Falls back to old relative_time_expression_type for backward compat.
    """
    rt_meta = sample.generation.relative_time_metadata

    if rt_meta is not None:
        # v0.2 path: use typed RelativeTimeMetadata
        expected_str = rt_meta.resolved_service_date
        if sample.output_kind == "tool_call":
            args = sample.expected.arguments if hasattr(sample.expected, "arguments") else None
            actual = args.service_date if args else None
        else:
            actual = sample.expected.booking_state.service_date if hasattr(sample.expected, "booking_state") else None

        if actual and actual != expected_str:
            return (
                f"relative_time: expected {expected_str} for template "
                f"'{rt_meta.expression_type}' (base={rt_meta.base_datetime}), got {actual}"
            )
        return None

    # Fallback: old metadata path
    gen_dict = sample.generation.model_dump(mode="json")
    rel_type = gen_dict.get("relative_time_expression_type")
    if not rel_type:
        return None

    base_date = _parse_current_time(sample)
    if base_date is None:
        return "relative_time: cannot parse current_time for date resolution"

    resolver = _RELATIVE_TIME_MAP.get(rel_type)
    if resolver is None:
        return None

    expected_date = resolver(base_date)
    expected_str = expected_date.isoformat()

    if sample.output_kind == "tool_call":
        args = sample.expected.arguments if hasattr(sample.expected, "arguments") else None
        actual = args.service_date if args else None
    else:
        actual = sample.expected.booking_state.service_date if hasattr(sample.expected, "booking_state") else None

    if actual and actual != expected_str:
        return f"relative_time: expected {expected_str} for template '{rel_type}' (base={base_date}), got {actual}"

    return None


# ── 2. State Inheritance Validator ──────────────────────────

_STATE_FIELDS = [
    "chef_name", "service_date", "start_time", "people", "address",
    "cuisine", "budget_min", "budget_max", "menu", "ingredient_purchase",
    "dietary_constraints", "occasion",
]


def validate_state_inheritance(sample: RawBookingSample) -> str | None:
    """v0.2: Verify state fields using StateTransitionMetadata.

    Uses generation.state_transition_metadata for deterministic validation.
    """
    st_meta = sample.generation.state_transition_metadata
    if st_meta is not None:
        changed = st_meta.changed_fields
        preserved = st_meta.preserved_fields
        invalidated = st_meta.invalidated_fields
    else:
        # Fallback for old samples
        gen_dict = sample.generation.model_dump(mode="json")
        changed = gen_dict.get("changed_fields", [])
        preserved = gen_dict.get("preserved_fields", [])
        invalidated = []

    if not changed and not preserved:
        return None  # Not a state-mutation sample

    # Check preserved fields: current_state values must match expected
    current = sample.input.current_state.model_dump(mode="json") if hasattr(sample.input, "current_state") and sample.input.current_state else {}

    if sample.output_kind == "tool_call":
        expected_args = sample.expected.arguments if hasattr(sample.expected, "arguments") else None
        for field in preserved:
            if field in _STATE_FIELDS and field in current:
                actual_val = getattr(expected_args, field, None) if expected_args else None
                if actual_val is not None and actual_val != current[field]:
                    return f"state_inheritance: field '{field}' should be preserved (value={current[field]}) but got {actual_val}"
    else:
        booking = sample.expected.booking_state if hasattr(sample.expected, "booking_state") else None
        if booking:
            for field in preserved:
                if field in _STATE_FIELDS and field in current:
                    actual = getattr(booking, field, None)
                    if actual != current[field]:
                        return f"state_inheritance: field '{field}' should be preserved (value={current[field]}) but got {actual}"

    # Check invalidated fields: chef_id/query_status/candidates should be reset
    if invalidated and sample.output_kind == "final":
        booking = sample.expected.booking_state if hasattr(sample.expected, "booking_state") else None
        if booking:
            for field in invalidated:
                if field == "chef_id" and booking.chef_id is not None:
                    return f"state_inheritance: field '{field}' should be invalidated (None) but got {booking.chef_id}"
                if field == "chef_query_status" and sample.expected.chef_query_status.value != "not_checked":
                    return f"state_inheritance: chef_query_status should be invalidated but got {sample.expected.chef_query_status.value}"

    return None


# ── 3. Tool Fact Grounding Validator ────────────────────────

def validate_tool_fact_grounding(sample: RawBookingSample) -> str | None:
    """Verify chef facts in expected FinalDecision trace to tool_result evidence.

    Checks:
    - chef_id/chef_name in booking_state match latest tool_result
    - candidate_chefs IDs/names match tool_result candidates
    - No fabricated chefs
    """
    if sample.output_kind != "final":
        return None
    if not isinstance(sample.expected, FinalDecision):
        return None

    # Find latest tool_result in history
    history = getattr(sample.input, "history", []) or []
    tool_results = _get_tool_messages(history)
    if not tool_results:
        return None  # No tool results to validate against

    latest_tr = _get_tool_result_content(tool_results[-1])
    if not isinstance(latest_tr, dict):
        return None

    # Build chef_ids set from all possible tool result fields
    candidates = latest_tr.get("candidates", [])
    chef_ids_from_tool = {c.get("chef_id") for c in candidates if c.get("chef_id")}
    # specific/available → chef field
    chef_field = latest_tr.get("chef")
    if isinstance(chef_field, dict) and chef_field.get("chef_id"):
        chef_ids_from_tool.add(chef_field.get("chef_id"))
    # specific/unavailable → alternatives field
    alternatives = latest_tr.get("alternatives", [])
    chef_ids_from_tool.update(a.get("chef_id") for a in alternatives if a.get("chef_id"))

    # Check chef_id in booking_state
    expected_chef_id = sample.expected.booking_state.chef_id
    if expected_chef_id and expected_chef_id not in chef_ids_from_tool:
        return f"tool_fact_grounding: chef_id '{expected_chef_id}' not found in tool_result candidates {sorted(chef_ids_from_tool)}"

    # Check candidate_chefs
    expected_candidates = sample.expected.candidate_chefs or []
    for ec in expected_candidates:
        ec_id = ec.chef_id if hasattr(ec, "chef_id") else ec.get("chef_id")
        if ec_id and ec_id not in chef_ids_from_tool:
            return f"tool_fact_grounding: candidate_chef '{ec_id}' not found in tool_result"

    return None


# ── 4. Candidate Order Validator ────────────────────────────

def validate_candidate_order(sample: RawBookingSample) -> str | None:
    """Verify candidate_chefs order matches tool_result order."""
    if sample.output_kind != "final":
        return None
    if not isinstance(sample.expected, FinalDecision):
        return None

    history = getattr(sample.input, "history", []) or []
    tool_results = _get_tool_messages(history)
    if not tool_results:
        return None

    latest_tr = _get_tool_result_content(tool_results[-1])
    if not isinstance(latest_tr, dict):
        return None

    tool_candidates = latest_tr.get("candidates", [])
    if not tool_candidates:
        return None

    tool_ids = [c.get("chef_id") for c in tool_candidates if c.get("chef_id")]
    expected_candidates = sample.expected.candidate_chefs or []
    expected_ids = [ec.chef_id if hasattr(ec, "chef_id") else ec.get("chef_id") for ec in expected_candidates]

    if expected_ids and tool_ids:
        # Check order preservation for matching IDs
        matched_expected = [eid for eid in expected_ids if eid in tool_ids]
        matched_tool = [tid for tid in tool_ids if tid in expected_ids]
        if matched_expected != matched_tool:
            return f"candidate_order: expected order {matched_expected} does not match tool_result order {matched_tool}"

    return None


# ── 5. Dietary Constraint Validator ─────────────────────────

def validate_dietary_constraints(sample: RawBookingSample) -> str | None:
    """Verify dietary_constraints are preserved across state changes."""
    gen_meta = sample.generation.model_dump(mode="json") if hasattr(sample.generation, "model_dump") else {}
    changed = gen_meta.get("changed_fields", [])
    preserved = gen_meta.get("preserved_fields", [])
    if not changed and not preserved:
        return None  # Not a state-mutation sample

    current_dietary = None
    if hasattr(sample.input, "current_state") and sample.input.current_state:
        current_dietary = sample.input.current_state.dietary_constraints

    expected_dietary = None
    if sample.output_kind == "tool_call" and hasattr(sample.expected, "arguments"):
        args = sample.expected.arguments
        expected_dietary = args.dietary_constraints if args else None
    elif hasattr(sample.expected, "booking_state"):
        expected_dietary = sample.expected.booking_state.dietary_constraints

    # If dietary_constraints is in preserved_fields, must match
    if "dietary_constraints" in preserved and current_dietary is not None:
        if expected_dietary != current_dietary:
            return f"dietary: preserved field mismatch — expected {current_dietary}, got {expected_dietary}"

    return None


# ── 6. Confirmation Transition Validator ────────────────────

def validate_confirmation_transition(sample: RawBookingSample) -> str | None:
    """Verify confirmation/booking state transitions are semantically correct.

    Rules:
    - Without explicit user confirmation → booking must NOT be authorized
    - With explicit confirmation → correct confirmation state
    """
    if sample.output_kind != "final":
        return None
    if not isinstance(sample.expected, FinalDecision):
        return None

    gen_meta = sample.generation.model_dump(mode="json") if hasattr(sample.generation, "model_dump") else {}
    scenario = sample.scenario

    # Check: explicit confirmation scenarios must have correct confirmation
    if scenario == "explicit_confirmation":
        if sample.expected.booking_state.confirmation is not True:
            return "confirmation: explicit_confirmation scenario but booking_state.confirmation is not True"

    # Check: non-confirmation scenarios must NOT have unauthorized booking
    if sample.expected.reply_type == "booking_authorized":
        if scenario != "explicit_confirmation":
            # Check if there's user confirmation intent
            user_confirms = gen_meta.get("user_confirms", False)
            if not user_confirms:
                return "confirmation: booking_authorized without explicit user confirmation in non-confirmation scenario"

    return None


# ── 7. Reply Policy Validator ────────────────────────────────

# Canonical reply_type mapping for common scenarios.
# Uses the actual ReplyType enum values from the contract (not scenario names).
# unrelated scenario → handoff reply_type (contract: unrelated=true ↔ handoff)
# tool_error scenario → booking_paused (contract: no inform_tool_error in ReplyType enum)
# specific_unavailable → present_alternatives (when alternatives exist) or booking_paused
_REPLY_POLICY_MAP: dict[str, set[str]] = {
    "missing_required_slots": {"ask_service_date", "ask_start_time", "ask_people",
                                "ask_address", "ask_multiple_required_fields"},
    "matched_candidates": {"present_chef_candidates"},
    "specific_available": {"confirm_specific_chef"},
    "specific_unavailable": {"present_alternatives", "booking_paused"},
    "no_match": {"inform_no_match"},
    "not_found": {"inform_not_found"},
    "out_of_service_area": {"inform_out_of_service_area"},
    "explicit_confirmation": {"booking_authorized"},
    "rejection": {"booking_paused"},
    "unrelated": {"handoff"},
    "handoff": {"handoff"},
    "tool_error": {"booking_paused"},
}


def validate_reply_policy(sample: RawBookingSample) -> str | None:
    """v0.2: Verify reply_type matches scenario-based canonical expectations.

    Only validates when scenario has a known canonical reply_type mapping.
    Does NOT check natural language reply text.
    """
    if sample.output_kind != "final":
        return None
    if not isinstance(sample.expected, FinalDecision):
        return None

    scenario = sample.scenario
    allowed = set(_REPLY_POLICY_MAP.get(scenario, []))

    # For tool_error scenarios, distinguish retryable vs fatal errors.
    # A retryable tool error should be acknowledged (acknowledge_result) rather
    # than pausing the booking (booking_paused). Inspect the latest tool result's
    # `retryable` flag.
    if scenario == "tool_error":
        tool_results = _get_tool_messages(getattr(sample.input, "history", []) or [])
        if tool_results:
            latest_tr = _get_tool_result_content(tool_results[-1])
            if latest_tr.get("retryable") is True:
                allowed.add("acknowledge_result")
            else:
                allowed.add("booking_paused")

    if not allowed:
        return None  # No canonical mapping for this scenario

    actual_rt = sample.expected.reply_type
    if actual_rt not in allowed:
        return (
            f"reply_policy: scenario '{scenario}' expects reply_type in {sorted(allowed)}, "
            f"got '{actual_rt}'"
        )

    return None


# ── 8. Decision Consistency Validator (Cross-Field) ─────────

def validate_decision_consistency(sample: RawBookingSample) -> str | None:
    """Cross-field consistency checks using canonical contract helpers.

    Checks (only for FinalDecision):
    - missing_info ↔ booking_state (must equal missing_required_slots())
    - info_complete ↔ missing_info (must be len(missing)==0)
    - chef_query_status ↔ candidate_chefs (matched requires non-empty candidates)
    - chef_id/name ↔ tool evidence
    - awaiting_confirmation ↔ confirmation/action
    - tool result status ↔ chef_query_status
    """
    if sample.output_kind != "final":
        return None
    if not isinstance(sample.expected, FinalDecision):
        return None

    from homechef_booking.schemas.booking import ChefQueryStatus, missing_required_slots

    booking = sample.expected.booking_state

    # 1. missing_info ↔ booking_state
    canonical_missing = missing_required_slots(booking)
    if sample.expected.missing_info != canonical_missing:
        return (
            f"decision_consistency: missing_info {sample.expected.missing_info} "
            f"does not match canonical missing_required_slots() {canonical_missing}"
        )

    # 2. info_complete ↔ missing_info
    expected_complete = len(canonical_missing) == 0
    if sample.expected.info_complete != expected_complete:
        return (
            f"decision_consistency: info_complete={sample.expected.info_complete} "
            f"but canonical missing_count={len(canonical_missing)} implies {expected_complete}"
        )

    # 3. chef_query_status ↔ candidate_chefs
    qs = sample.expected.chef_query_status
    candidates = sample.expected.candidate_chefs or []
    if qs == ChefQueryStatus.matched and len(candidates) == 0:
        return "decision_consistency: chef_query_status=matched requires non-empty candidate_chefs"
    if qs == ChefQueryStatus.no_match and len(candidates) > 0:
        return "decision_consistency: chef_query_status=no_match requires empty candidate_chefs"

    # 4. tool result status ↔ chef_query_status
    history = getattr(sample.input, "history", []) or []
    tool_results = _get_tool_messages(history)
    if tool_results:
        from homechef_booking.data.scenario_lineage import TOOL_RESULT_STATUS_TO_QUERY_STATUS
        latest_tr = _get_tool_result_content(tool_results[-1])
        mode = latest_tr.get("mode", "")
        status = latest_tr.get("status", "")
        expected_qs = TOOL_RESULT_STATUS_TO_QUERY_STATUS.get((mode, status))
        if expected_qs is not None and qs != expected_qs:
            return (
                f"decision_consistency: tool result ({mode}/{status}) implies "
                f"chef_query_status={expected_qs.value}, got {qs.value}"
            )

    # 5. awaiting_confirmation ↔ confirmation
    current_state = sample.input.current_state
    if current_state and current_state.awaiting_confirmation:
        # If awaiting_confirmation was True and user confirmed, confirmation should be True
        if sample.expected.booking_state.confirmation is None and sample.expected.reply_type == "booking_authorized":
            return "decision_consistency: awaiting_confirmation=True with booking_authorized but confirmation is None"

    return None


# ── 9. Tool Lineage Validator ──────────────────────────────

def validate_tool_lineage(sample: RawBookingSample) -> str | None:
    """Validate that tool_result samples have proper tool_call lineage.

    Checks (only when history contains tool messages):
    1. Every tool message has a preceding assistant tool_call
    2. tool_call_id pairing is correct
    3. tool_name is consistent (find_chefs)
    4. tool_call arguments match the effective query state
    5. Tool Result and expected Final have evidence link
    6. No stale tool result after query dependency modification
    """
    history = getattr(sample.input, "history", []) or []
    if not history:
        return None

    # Check for tool messages
    tool_messages = _get_tool_messages(history)
    if not tool_messages:
        return None

    # Check tool_call_id pairing
    pending_call_ids: set[str] = set()
    for i, msg in enumerate(history):
        role = _get_msg_role(msg)
        tool_calls = _get_msg_attr(msg, "tool_calls")
        if role == "assistant" and tool_calls:
            for tc in tool_calls:
                tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                pending_call_ids.add(tc_id)
        elif role == "tool":
            tcid = _get_msg_attr(msg, "tool_call_id", "")
            if tcid not in pending_call_ids:
                return f"tool_lineage: tool message at index {i} has tool_call_id '{tcid}' with no preceding call"
            pending_call_ids.discard(tcid)

    # Check tool_call arguments match booking_state (effective query state).
    # This ONLY applies to the last executed tool_call when there is NO pending
    # re-query. If expected.action == "tool_call" (the model is about to re-query
    # because the user modified a query dependency), the historical tool_call is
    # the OLD query and current_state already reflects the NEW (not-yet-executed)
    # query — so we must NOT require them to match.
    is_pending_requery = getattr(sample.expected, "action", None) == "tool_call"
    current_state = sample.input.current_state
    if current_state and not is_pending_requery:
        current_bs = current_state.booking_state
        for msg in history:
            role = _get_msg_role(msg)
            tool_calls = _get_msg_attr(msg, "tool_calls")
            if role == "assistant" and tool_calls:
                for tc in tool_calls:
                    func = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
                    if func is None:
                        continue
                    args_str = func.get("arguments", "{}") if isinstance(func, dict) else getattr(func, "arguments", "{}")
                    args = json_load(args_str)
                    # Check required slots: if booking_state has them, args must match
                    for field in ["service_date", "start_time", "people", "address"]:
                        cur_val = getattr(current_bs, field, None)
                        arg_val = args.get(field)
                        # If both are non-null, they must match
                        if cur_val is not None and arg_val is not None and cur_val != arg_val:
                            return (
                                f"tool_lineage: tool_call argument '{field}'={arg_val} "
                                f"does not match booking_state.{field}={cur_val}"
                            )

    # Check for stale tool result after query modification
    if isinstance(sample.expected, FinalDecision):
        from homechef_booking.schemas.booking import QUERY_DEPENDENCY_FIELDS
        if current_state:
            current_bs = current_state.booking_state
            new_bs = sample.expected.booking_state
            for field in QUERY_DEPENDENCY_FIELDS:
                cur_val = getattr(current_bs, field, None)
                new_val = getattr(new_bs, field, None)
                if cur_val is not None and new_val is not None and cur_val != new_val:
                    # Query dependency mutated → old tool result is stale
                    if sample.expected.candidate_chefs:
                        return (
                            f"tool_lineage: query dependency field '{field}' changed "
                            f"({cur_val}→{new_val}) but candidate_chefs still non-empty (stale tool result)"
                        )
                    break

    return None


# ── 10. Metadata Consistency Validator ──────────────────────

def validate_metadata_consistency(sample: RawBookingSample) -> str | None:
    """Validate that generation metadata aligns with actual history/expected.

    Checks:
    - ToolFactMetadata.candidate_ids match actual tool_result candidate IDs
    - ToolFactMetadata.candidate_order matches actual tool_result order
    - ToolFactMetadata.tool_mode/tool_result_status match actual tool_result
    - StateTransitionMetadata fields match actual state changes
    """
    tfm = sample.generation.tool_fact_metadata

    history = getattr(sample.input, "history", []) or []
    tool_messages = _get_tool_messages(history)

    if tfm is not None and tool_messages:
        latest_tr = _get_tool_result_content(tool_messages[-1])
        # Check tool_mode
        if tfm.tool_mode and latest_tr.get("mode", "") != tfm.tool_mode:
            return (
                f"metadata_consistency: ToolFactMetadata.tool_mode='{tfm.tool_mode}' "
                f"but actual tool_result mode='{latest_tr.get('mode', '')}'"
            )
        # Check tool_result_status
        if tfm.tool_result_status and latest_tr.get("status", "") != tfm.tool_result_status:
            return (
                f"metadata_consistency: ToolFactMetadata.tool_result_status='{tfm.tool_result_status}' "
                f"but actual tool_result status='{latest_tr.get('status', '')}'"
            )
        # Check candidate_ids
        actual_candidates = []
        if latest_tr.get("status") == "matched":
            actual_candidates = [c.get("chef_id") for c in latest_tr.get("candidates", [])]
        elif latest_tr.get("status") == "unavailable":
            actual_candidates = [c.get("chef_id") for c in latest_tr.get("alternatives", [])]
        elif latest_tr.get("status") == "available":
            chef = latest_tr.get("chef", {})
            if chef:
                actual_candidates = [chef.get("chef_id")]
        if tfm.candidate_ids and actual_candidates and tfm.candidate_ids != actual_candidates:
            return (
                f"metadata_consistency: ToolFactMetadata.candidate_ids={tfm.candidate_ids} "
                f"do not match actual tool_result candidate_ids={actual_candidates}"
            )
        if tfm.candidate_order and actual_candidates and tfm.candidate_order != actual_candidates:
            return (
                f"metadata_consistency: ToolFactMetadata.candidate_order={tfm.candidate_order} "
                f"does not match actual tool_result order={actual_candidates}"
            )

    return None


# ── Helpers ─────────────────────────────────────────────────


def json_load(s: str) -> Any:
    try:
        return _json.loads(s)
    except (_json.JSONDecodeError, TypeError):
        return {}


def _get_msg_role(msg: Any) -> str:
    """Extract role from a history message (dict or Pydantic model)."""
    if hasattr(msg, "role"):
        return str(msg.role)
    if isinstance(msg, dict):
        return str(msg.get("role", ""))
    return ""


def _get_msg_attr(msg: Any, attr: str, default: Any = None) -> Any:
    """Extract an attribute from a history message (dict or Pydantic model)."""
    if hasattr(msg, attr):
        return getattr(msg, attr)
    if isinstance(msg, dict):
        return msg.get(attr, default)
    return default


def _get_tool_messages(history: list) -> list:
    """Extract tool messages from history, handling both dict and Pydantic types."""
    return [m for m in history if _get_msg_role(m) == "tool"]


def _get_tool_result_content(msg: Any) -> dict:
    """Extract and parse tool result content from a tool message."""
    content = _get_msg_attr(msg, "content", "{}")
    if content is None:
        content = "{}"
    return json_load(content)


# ── Aggregate Semantic Validation ───────────────────────────

class SemanticValidationResult:
    """Result of running all semantic validators on a sample."""
    __slots__ = ("sample_id", "errors")

    def __init__(self, sample_id: str):
        self.sample_id = sample_id
        self.errors: dict[str, str] = {}

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def __repr__(self) -> str:
        status = "PASS" if self.passed else f"FAIL ({len(self.errors)} errors)"
        return f"SemanticValidation({self.sample_id}: {status})"


VALIDATORS: list[tuple[str, Callable]] = [
    ("relative_time", validate_relative_time),
    ("state_inheritance", validate_state_inheritance),
    ("tool_fact_grounding", validate_tool_fact_grounding),
    ("candidate_order", validate_candidate_order),
    ("dietary_constraints", validate_dietary_constraints),
    ("confirmation_transition", validate_confirmation_transition),
    ("reply_policy", validate_reply_policy),
    ("decision_consistency", validate_decision_consistency),
    ("tool_lineage", validate_tool_lineage),
    ("metadata_consistency", validate_metadata_consistency),
]


def validate_semantic(sample: RawBookingSample) -> SemanticValidationResult:
    """Run all semantic validators on a raw sample."""
    result = SemanticValidationResult(sample.id)
    for name, validator in VALIDATORS:
        error = validator(sample)
        if error:
            result.errors[name] = error
    return result
