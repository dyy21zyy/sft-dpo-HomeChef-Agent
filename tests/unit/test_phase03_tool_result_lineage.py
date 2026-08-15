"""TDD for Phase 03 v0.2 Tool Result Scenario Lineage.

Tests cover:
  1.  search matched: query state complete, tool call consistent, candidates consistent
  2.  search no_match
  3.  specific available
  4.  specific unavailable
  5.  specific not_found
  6.  tool error
  7.  out_of_service_area
  8.  matched + empty candidate rejected
  9.  candidate not present in tool result rejected
  10. tool arguments inconsistent with booking/query state rejected
  11. stale tool result after query modification rejected
  12. missing_info inconsistent with booking state rejected
"""

from __future__ import annotations

import json

from homechef_booking.data.raw_sample import RawBookingSample
from homechef_booking.data.raw_validator import validate_raw_sample
from homechef_booking.data.scenario_lineage import (
    TOOL_RESULT_STATUS_TO_QUERY_STATUS,
    TOOL_RESULT_STATUS_TO_REPLY_TYPE,
    ScenarioFacts,
    build_current_state,
    build_expected_final,
    build_expected_tool_call,
    build_find_chefs_arguments,
    build_history_with_tool_result,
    derive_query_state_from_tool_result,
)
from homechef_booking.data.semantic_validators import (
    validate_decision_consistency,
    validate_semantic,
    validate_tool_lineage,
)
from homechef_booking.data.tool_spec_factory import canonical_find_chefs_tool_dict
from homechef_booking.schemas.booking import (
    BookingSlot,
    ChefQueryStatus,
    ReplyType,
)

# ── Fixtures ─────────────────────────────────────────────────


def _complete_booking() -> BookingSlot:
    return BookingSlot(
        service_date="2026-08-15",
        start_time="18:00",
        people=4,
        address="Beijing",
        cuisine="Sichuan",
    )


def _make_raw_sample(
    facts: ScenarioFacts,
    user_input: str = "test",
    output_kind: str = "final",
    history: list | None = None,
    current_state: dict | None = None,
    expected: dict | None = None,
    scenario: str = "tool_result",
) -> RawBookingSample:
    """Build a RawBookingSample from ScenarioFacts for testing."""
    if history is None:
        history = build_history_with_tool_result(user_input, facts) if facts.has_tool_result else []
    if current_state is None:
        current_state = build_current_state(facts)
    if expected is None:
        if output_kind == "tool_call":
            expected = build_expected_tool_call(facts)
        else:
            expected = build_expected_final(facts)

    gen_meta = {
        "generator": "deterministic_smoke_v02",
        "model": "smoke_v02",
        "seed": 9999,
        "prompt_sha256": "a" * 64,
        "generated_at": "2026-08-12T00:00:00Z",
        "scenario": scenario,
        "capability_tags": [],
        "template_id": "",
    }
    if facts.tool_result_payload is not None:
        from homechef_booking.data.raw_sample import ToolFactMetadata
        cids = [c.chef_id for c in facts.effective_candidates]
        gen_meta["tool_fact_metadata"] = ToolFactMetadata(
            tool_mode=facts.tool_mode,
            tool_result_status=facts.tool_result_status,
            requested_chef=facts.requested_chef_name,
            candidate_ids=cids,
            candidate_order=cids,
            evidence_fields=["chef_id", "chef_name"],
        ).model_dump()

    return RawBookingSample.model_validate({
        "id": "test-lineage-001",
        "dataset_version": "phase03_v0.2",
        "contract_id": "homechef-booking-v1",
        "source": "synthetic",
        "scenario": scenario,
        "output_kind": output_kind,
        "conversation_kind": "multi_turn" if history else "single_turn",
        "tags": [],
        "input": {
            "history": history,
            "current_state": current_state,
            "user_input": user_input,
            "current_time": "2026-08-12 18:00",
            "available_tools": [canonical_find_chefs_tool_dict()] if output_kind == "tool_call" else [],
        },
        "expected": expected,
        "generation": gen_meta,
        "review": {"status": "machine_validated"},
        "dpo_targets": [],
    })


# ── TEST 1: search matched ──────────────────────────────────


def test_search_matched_lineage():
    """TEST 1: search/matched — query state complete, tool call consistent, candidates consistent."""
    bs = _complete_booking()
    candidates = [
        {"chef_id": "C001", "chef_name": "Chef A"},
        {"chef_id": "C002", "chef_name": "Chef B"},
    ]
    tool_result = {"mode": "search", "status": "matched", "candidates": candidates}
    facts = ScenarioFacts(
        booking_state=bs,
        tool_mode="search",
        tool_result_status="matched",
        tool_result_payload=tool_result,
        scenario="tool_result",
    )

    # Verify deterministic derivation
    assert facts.effective_query_status == ChefQueryStatus.matched
    assert len(facts.effective_candidates) == 2
    assert facts.effective_candidates[0].chef_id == "C001"
    assert facts.effective_info_complete is True
    assert facts.effective_missing_info == []
    assert facts.effective_reply_type == ReplyType.present_chef_candidates

    # Build and validate sample
    sample = _make_raw_sample(facts, user_input="found chefs")
    errors = validate_raw_sample(sample)
    assert errors == [], f"Schema errors: {[e.message for e in errors]}"

    sem = validate_semantic(sample)
    assert sem.passed, f"Semantic errors: {sem.errors}"

    # Tool lineage must pass
    assert validate_tool_lineage(sample) is None
    assert validate_decision_consistency(sample) is None


# ── TEST 2: search no_match ─────────────────────────────────


def test_search_no_match_lineage():
    """TEST 2: search/no_match — empty candidates, inform_no_match."""
    bs = _complete_booking()
    tool_result = {"mode": "search", "status": "no_match", "candidates": []}
    facts = ScenarioFacts(
        booking_state=bs,
        tool_mode="search",
        tool_result_status="no_match",
        tool_result_payload=tool_result,
        scenario="tool_result",
    )

    assert facts.effective_query_status == ChefQueryStatus.no_match
    assert facts.effective_candidates == []
    assert facts.effective_reply_type == ReplyType.inform_no_match
    assert facts.effective_info_complete is True

    sample = _make_raw_sample(facts, user_input="no chefs?")
    errors = validate_raw_sample(sample)
    assert errors == [], f"Schema errors: {[e.message for e in errors]}"
    sem = validate_semantic(sample)
    assert sem.passed, f"Semantic errors: {sem.errors}"


# ── TEST 3: specific available ──────────────────────────────


def test_specific_available_lineage():
    """TEST 3: specific/available — chef found, confirm_specific_chef."""
    bs = _complete_booking()
    bs.chef_name = "Chef Zhang"
    chef = {"chef_id": "C005", "chef_name": "Chef Zhang"}
    tool_result = {"mode": "specific", "status": "available", "chef": chef}
    facts = ScenarioFacts(
        booking_state=bs,
        tool_mode="specific",
        tool_result_status="available",
        tool_result_payload=tool_result,
        requested_chef_name="Chef Zhang",
        scenario="tool_result",
    )

    assert facts.effective_query_status == ChefQueryStatus.available
    assert facts.effective_chef_id == "C005"
    assert facts.effective_chef_name == "Chef Zhang"
    assert facts.effective_reply_type == ReplyType.confirm_specific_chef

    sample = _make_raw_sample(facts, user_input="available?")
    errors = validate_raw_sample(sample)
    assert errors == [], f"Schema errors: {[e.message for e in errors]}"
    sem = validate_semantic(sample)
    assert sem.passed, f"Semantic errors: {sem.errors}"


# ── TEST 4: specific unavailable ────────────────────────────


def test_specific_unavailable_lineage():
    """TEST 4: specific/unavailable — alternatives presented."""
    bs = _complete_booking()
    bs.chef_name = "Chef Zhang"
    alternatives = [
        {"chef_id": "C006", "chef_name": "Chef Chen"},
    ]
    tool_result = {
        "mode": "specific", "status": "unavailable",
        "requested_chef": "Chef Zhang", "alternatives": alternatives,
    }
    facts = ScenarioFacts(
        booking_state=bs,
        tool_mode="specific",
        tool_result_status="unavailable",
        tool_result_payload=tool_result,
        requested_chef_name="Chef Zhang",
        scenario="tool_result",
    )

    assert facts.effective_query_status == ChefQueryStatus.unavailable
    assert len(facts.effective_candidates) == 1
    assert facts.effective_candidates[0].chef_id == "C006"
    assert facts.effective_reply_type == ReplyType.present_alternatives

    sample = _make_raw_sample(facts, user_input="alternatives?")
    errors = validate_raw_sample(sample)
    assert errors == [], f"Schema errors: {[e.message for e in errors]}"
    sem = validate_semantic(sample)
    assert sem.passed, f"Semantic errors: {sem.errors}"


# ── TEST 5: specific not_found ──────────────────────────────


def test_specific_not_found_lineage():
    """TEST 5: specific/not_found — inform_not_found."""
    bs = _complete_booking()
    bs.chef_name = "Chef Ghost"
    tool_result = {
        "mode": "specific", "status": "not_found",
        "requested_chef": "Chef Ghost", "alternatives": [],
    }
    facts = ScenarioFacts(
        booking_state=bs,
        tool_mode="specific",
        tool_result_status="not_found",
        tool_result_payload=tool_result,
        requested_chef_name="Chef Ghost",
        scenario="tool_result",
    )

    assert facts.effective_query_status == ChefQueryStatus.not_found
    assert facts.effective_candidates == []
    assert facts.effective_reply_type == ReplyType.inform_not_found

    sample = _make_raw_sample(facts, user_input="not found?")
    errors = validate_raw_sample(sample)
    assert errors == [], f"Schema errors: {[e.message for e in errors]}"
    sem = validate_semantic(sample)
    assert sem.passed, f"Semantic errors: {sem.errors}"


# ── TEST 6: tool error ──────────────────────────────────────


def test_tool_error_lineage():
    """TEST 6: search/error — booking_paused."""
    bs = _complete_booking()
    tool_result = {
        "mode": "search", "status": "error",
        "error_code": "SERVICE_ERROR", "retryable": False,
        "message": "service unavailable",
    }
    facts = ScenarioFacts(
        booking_state=bs,
        tool_mode="search",
        tool_result_status="error",
        tool_result_payload=tool_result,
        scenario="tool_error",
    )

    assert facts.effective_query_status == ChefQueryStatus.error
    assert facts.effective_candidates == []
    assert facts.effective_reply_type == ReplyType.booking_paused

    sample = _make_raw_sample(facts, user_input="error?")
    errors = validate_raw_sample(sample)
    assert errors == [], f"Schema errors: {[e.message for e in errors]}"
    sem = validate_semantic(sample)
    assert sem.passed, f"Semantic errors: {sem.errors}"


# ── TEST 7: out_of_service_area ─────────────────────────────


def test_out_of_service_area_lineage():
    """TEST 7: search/out_of_service_area — inform_out_of_service_area."""
    bs = _complete_booking()
    tool_result = {"mode": "search", "status": "out_of_service_area", "candidates": []}
    facts = ScenarioFacts(
        booking_state=bs,
        tool_mode="search",
        tool_result_status="out_of_service_area",
        tool_result_payload=tool_result,
        scenario="tool_result",
    )

    assert facts.effective_query_status == ChefQueryStatus.out_of_service_area
    assert facts.effective_candidates == []
    assert facts.effective_reply_type == ReplyType.inform_out_of_service_area

    sample = _make_raw_sample(facts, user_input="service area?")
    errors = validate_raw_sample(sample)
    assert errors == [], f"Schema errors: {[e.message for e in errors]}"
    sem = validate_semantic(sample)
    assert sem.passed, f"Semantic errors: {sem.errors}"


# ── TEST 8: matched + empty candidate rejected ──────────────


def test_matched_with_empty_candidates_rejected():
    """TEST 8: chef_query_status=matched with empty candidate_chefs must fail."""
    bs = _complete_booking()
    tool_result = {"mode": "search", "status": "matched", "candidates": [
        {"chef_id": "C001", "chef_name": "Chef A"},
    ]}
    facts = ScenarioFacts(
        booking_state=bs,
        tool_mode="search",
        tool_result_status="matched",
        tool_result_payload=tool_result,
        scenario="tool_result",
    )
    sample = _make_raw_sample(facts)

    # Tamper: set candidate_chefs to empty but keep qs=matched
    tampered_expected = dict(sample.expected.model_dump(mode="json"))
    tampered_expected["candidate_chefs"] = []
    tampered_sample = sample.model_copy(deep=True)
    tampered_sample = RawBookingSample.model_validate({
        **sample.model_dump(mode="json"),
        "expected": tampered_expected,
    })

    # decision_consistency must catch this
    err = validate_decision_consistency(tampered_sample)
    assert err is not None
    assert "matched requires non-empty candidate_chefs" in err


# ── TEST 9: candidate not present in tool result rejected ───


def test_candidate_not_in_tool_result_rejected():
    """TEST 9: candidate_chefs containing a chef_id not in tool_result must fail."""
    bs = _complete_booking()
    tool_result = {"mode": "search", "status": "matched", "candidates": [
        {"chef_id": "C001", "chef_name": "Chef A"},
    ]}
    facts = ScenarioFacts(
        booking_state=bs,
        tool_mode="search",
        tool_result_status="matched",
        tool_result_payload=tool_result,
        scenario="tool_result",
    )
    sample = _make_raw_sample(facts)

    # Tamper: add a fabricated candidate
    tampered_expected = dict(sample.expected.model_dump(mode="json"))
    tampered_expected["candidate_chefs"] = [
        {"chef_id": "C001", "chef_name": "Chef A"},
        {"chef_id": "FAKE999", "chef_name": "Fake Chef"},
    ]
    tampered_sample = RawBookingSample.model_validate({
        **sample.model_dump(mode="json"),
        "expected": tampered_expected,
    })

    # tool_fact_grounding must catch fabricated chef
    from homechef_booking.data.semantic_validators import validate_tool_fact_grounding
    err = validate_tool_fact_grounding(tampered_sample)
    assert err is not None
    assert "not found in tool_result" in err


# ── TEST 10: tool arguments inconsistent with booking state ─


def test_tool_args_inconsistent_with_booking_rejected():
    """TEST 10: tool_call arguments not matching booking_state must fail tool_lineage."""
    bs = _complete_booking()
    bs.service_date = "2026-08-15"
    facts = ScenarioFacts(
        booking_state=bs,
        tool_mode="search",
        tool_result_status="matched",
        tool_result_payload={"mode": "search", "status": "matched", "candidates": [
            {"chef_id": "C001", "chef_name": "Chef A"},
        ]},
        scenario="tool_result",
    )

    # Build history with WRONG arguments (date mismatch)
    args = build_find_chefs_arguments(facts)
    args["service_date"] = "2026-09-01"  # Wrong date
    history = [
        {"role": "user", "content": "find chefs"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_001", "type": "function", "function": {
                "name": "find_chefs",
                "arguments": json.dumps(args, ensure_ascii=False),
            }},
        ]},
        {"role": "tool", "tool_call_id": "call_001", "name": "find_chefs",
         "content": json.dumps(facts.tool_result_payload, ensure_ascii=False)},
    ]
    sample = _make_raw_sample(facts, history=history)

    err = validate_tool_lineage(sample)
    assert err is not None
    assert "service_date" in err
    assert "does not match" in err


# ── TEST 11: stale tool result after query modification ─────


def test_stale_tool_result_after_query_modification_rejected():
    """TEST 11: query dependency field changed but candidate_chefs still non-empty → stale."""
    bs = _complete_booking()
    tool_result = {"mode": "search", "status": "matched", "candidates": [
        {"chef_id": "C001", "chef_name": "Chef A"},
    ]}
    facts = ScenarioFacts(
        booking_state=bs,
        tool_mode="search",
        tool_result_status="matched",
        tool_result_payload=tool_result,
        scenario="tool_result",
    )
    sample = _make_raw_sample(facts)

    # Tamper: change service_date in expected but keep candidate_chefs
    tampered_expected = dict(sample.expected.model_dump(mode="json"))
    tampered_expected["booking_state"]["service_date"] = "2026-12-01"  # Changed
    tampered_sample = RawBookingSample.model_validate({
        **sample.model_dump(mode="json"),
        "expected": tampered_expected,
    })

    # tool_lineage should catch stale tool result
    err = validate_tool_lineage(tampered_sample)
    assert err is not None
    assert "stale tool result" in err or "query dependency" in err


# ── TEST 12: missing_info inconsistent with booking state ───


def test_missing_info_inconsistent_with_booking_rejected():
    """TEST 12: missing_info not matching missing_required_slots() must fail."""
    bs = BookingSlot()  # All None
    facts = ScenarioFacts(booking_state=bs, scenario="missing_required_slots")
    sample = _make_raw_sample(facts, output_kind="final", scenario="missing_required_slots")

    # Tamper: set missing_info to only ["service_date"] when all 4 are missing
    tampered_expected = dict(sample.expected.model_dump(mode="json"))
    tampered_expected["missing_info"] = ["service_date"]  # Should be all 4
    tampered_expected["info_complete"] = False
    tampered_sample = RawBookingSample.model_validate({
        **sample.model_dump(mode="json"),
        "expected": tampered_expected,
    })

    err = validate_decision_consistency(tampered_sample)
    assert err is not None
    assert "missing_info" in err
    assert "does not match canonical" in err


# ── Bonus: verify deterministic mapping table ────────────────


def test_tool_result_status_mapping_completeness():
    """Verify all 9 tool result (mode, status) pairs have deterministic mappings."""
    expected_pairs = {
        ("search", "matched"),
        ("search", "no_match"),
        ("search", "out_of_service_area"),
        ("search", "error"),
        ("specific", "available"),
        ("specific", "unavailable"),
        ("specific", "not_found"),
        ("specific", "out_of_service_area"),
        ("specific", "error"),
    }
    assert set(TOOL_RESULT_STATUS_TO_QUERY_STATUS.keys()) == expected_pairs
    assert set(TOOL_RESULT_STATUS_TO_REPLY_TYPE.keys()) == expected_pairs


def test_derive_query_state_from_tool_result_specific():
    """Verify derive_query_state_from_tool_result sets chef_name for specific mode."""
    bs = _complete_booking()
    tool_result = {
        "mode": "specific", "status": "unavailable",
        "requested_chef": "Chef Zhang", "alternatives": [],
    }
    derived = derive_query_state_from_tool_result(tool_result, bs)
    assert derived.chef_name == "Chef Zhang"
    assert derived.service_date == bs.service_date  # preserved
