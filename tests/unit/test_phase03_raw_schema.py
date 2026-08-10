"""Task 1 Phase 03: Raw sample schema tests."""

import json

from homechef_booking.data.raw_sample import RawBookingSample, parse_raw_sample_line

FIND_CHEFS_TOOL = {
    "type": "function",
    "function": {
        "name": "find_chefs",
        "description": "Find chefs",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["chef_name", "service_date", "start_time", "people", "address", "cuisine", "budget_min", "budget_max", "menu", "ingredient_purchase", "dietary_constraints", "occasion"],
            "properties": {
                "chef_name": {"type": ["string", "null"]},
                "service_date": {"type": ["string", "null"], "pattern": r"^\d{4}-\d{2}-\d{2}$", "format": "date"},
                "start_time": {"type": ["string", "null"], "pattern": r"^([01]\d|2[0-3]):[0-5]\d$"},
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
        },
    },
}


def empty_booking():
    return {
        "service_date": None, "start_time": None, "people": None, "address": None,
        "cuisine": None, "budget_min": None, "budget_max": None, "menu": [],
        "chef_id": None, "chef_name": None, "ingredient_purchase": None,
        "dietary_constraints": [], "occasion": None, "confirmation": None,
    }


def test_raw_sample_accepts_valid_contract_shapes():
    line = json.dumps({
        "id": "phase03-raw-000001",
        "dataset_version": "phase03_v0.1",
        "contract_id": "homechef-booking-v1",
        "source": "synthetic",
        "scenario": "missing_required_slots",
        "output_kind": "final",
        "conversation_kind": "single_turn",
        "tags": ["missing_required_slots"],
        "input": {
            "history": [],
            "current_state": {
                "booking_state": empty_booking(),
                "chef_query_status": "not_checked",
                "candidate_chefs": [],
                "awaiting_confirmation": False,
            },
            "user_input": "想约一个家宴",
            "current_time": "2026-08-09 18:00",
            "available_tools": [],
        },
        "expected": {
            "action": "final",
            "booking_state": {
                "service_date": None, "start_time": None, "people": None, "address": None,
                "cuisine": None, "budget_min": None, "budget_max": None, "menu": [],
                "chef_id": None, "chef_name": None, "ingredient_purchase": None,
                "dietary_constraints": [], "occasion": "家宴", "confirmation": None,
            },
            "chef_query_status": "not_checked",
            "candidate_chefs": [],
            "info_complete": False,
            "unrelated": False,
            "missing_info": ["service_date", "start_time", "people", "address"],
            "reply_type": "ask_multiple_required_fields",
            "reply": "请补充用餐日期、开始时间、人数和服务地址。",
        },
        "generation": {
            "generator": "openai_responses",
            "model": "gpt-5.6-sol",
            "seed": 3001,
            "prompt_sha256": "a" * 64,
            "generated_at": "2026-08-10T00:00:00Z",
        },
        "review": {"status": "machine_validated", "reviewer": None, "notes": []},
        "dpo_targets": ["H2"],
    }, ensure_ascii=False)
    sample = parse_raw_sample_line(line)
    assert isinstance(sample, RawBookingSample)
    assert sample.output_kind == sample.expected.action


def test_raw_sample_rejects_invalid_contract_id():
    line = json.dumps({
        "id": "phase03-raw-000001",
        "dataset_version": "phase03_v0.1",
        "contract_id": "bad-contract",
        "source": "synthetic",
        "scenario": "missing_required_slots",
        "output_kind": "final",
        "conversation_kind": "single_turn",
        "tags": [],
        "input": {
            "history": [],
            "current_state": {"booking_state": empty_booking(), "chef_query_status": "not_checked", "candidate_chefs": [], "awaiting_confirmation": False},
            "user_input": "想约一个家宴",
            "current_time": "2026-08-09 18:00",
            "available_tools": [],
        },
        "expected": {
            "action": "final",
            "booking_state": empty_booking(),
            "chef_query_status": "not_checked",
            "candidate_chefs": [],
            "info_complete": False,
            "unrelated": False,
            "missing_info": ["service_date"],
            "reply_type": "ask_service_date",
            "reply": "请提供日期",
        },
        "generation": {
            "generator": "openai_responses",
            "model": "gpt-5.6-sol",
            "seed": 3001,
            "prompt_sha256": "a" * 64,
            "generated_at": "2026-08-10T00:00:00Z",
        },
        "review": {"status": "machine_validated", "reviewer": None, "notes": []},
        "dpo_targets": [],
    }, ensure_ascii=False)
    try:
        parse_raw_sample_line(line)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_raw_sample_rejects_invalid_dpo_target():
    line = json.dumps({
        "id": "phase03-raw-000001",
        "dataset_version": "phase03_v0.1",
        "contract_id": "homechef-booking-v1",
        "source": "synthetic",
        "scenario": "missing_required_slots",
        "output_kind": "final",
        "conversation_kind": "single_turn",
        "tags": [],
        "input": {
            "history": [],
            "current_state": {"booking_state": empty_booking(), "chef_query_status": "not_checked", "candidate_chefs": [], "awaiting_confirmation": False},
            "user_input": "你好",
            "current_time": "2026-08-09 18:00",
            "available_tools": [],
        },
        "expected": {
            "action": "final",
            "booking_state": empty_booking(),
            "chef_query_status": "not_checked",
            "candidate_chefs": [],
            "info_complete": False,
            "unrelated": False,
            "missing_info": ["service_date"],
            "reply_type": "ask_service_date",
            "reply": "请提供日期",
        },
        "generation": {
            "generator": "openai_responses",
            "model": "gpt-5.6-sol",
            "seed": 3001,
            "prompt_sha256": "a" * 64,
            "generated_at": "2026-08-10T00:00:00Z",
        },
        "review": {"status": "machine_validated", "reviewer": None, "notes": []},
        "dpo_targets": ["H9"],
    }, ensure_ascii=False)
    try:
        parse_raw_sample_line(line)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_raw_sample_rejects_output_kind_mismatch():
    line = json.dumps({
        "id": "phase03-raw-000001",
        "dataset_version": "phase03_v0.1",
        "contract_id": "homechef-booking-v1",
        "source": "synthetic",
        "scenario": "valid_search_tool_call",
        "output_kind": "tool_call",
        "conversation_kind": "single_turn",
        "tags": [],
        "input": {
            "history": [],
            "current_state": {
                "booking_state": {"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0, "menu": ["水煮鱼"], "chef_id": None, "chef_name": None, "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日", "confirmation": None},
                "chef_query_status": "not_checked", "candidate_chefs": [], "awaiting_confirmation": False,
            },
            "user_input": "明晚六点两个人川菜不吃花生",
            "current_time": "2026-08-09 18:00",
            "available_tools": [FIND_CHEFS_TOOL],
        },
        "expected": {
            "action": "final",
            "booking_state": empty_booking(),
            "chef_query_status": "not_checked",
            "candidate_chefs": [],
            "info_complete": False,
            "unrelated": False,
            "missing_info": ["service_date"],
            "reply_type": "ask_service_date",
            "reply": "请提供日期",
        },
        "generation": {
            "generator": "openai_responses",
            "model": "gpt-5.6-sol",
            "seed": 3001,
            "prompt_sha256": "a" * 64,
            "generated_at": "2026-08-10T00:00:00Z",
        },
        "review": {"status": "machine_validated", "reviewer": None, "notes": []},
        "dpo_targets": [],
    }, ensure_ascii=False)
    try:
        parse_raw_sample_line(line)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_raw_sample_accepts_valid_tool_call_shape():
    line = json.dumps({
        "id": "phase03-raw-000002",
        "dataset_version": "phase03_v0.1",
        "contract_id": "homechef-booking-v1",
        "source": "synthetic",
        "scenario": "valid_search_tool_call",
        "output_kind": "tool_call",
        "conversation_kind": "single_turn",
        "tags": ["valid_search_tool_call"],
        "input": {
            "history": [],
            "current_state": {
                "booking_state": {"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0, "menu": ["水煮鱼"], "chef_id": None, "chef_name": None, "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日", "confirmation": None},
                "chef_query_status": "not_checked", "candidate_chefs": [], "awaiting_confirmation": False,
            },
            "user_input": "明晚六点两个人川菜不吃花生",
            "current_time": "2026-08-09 18:00",
            "available_tools": [FIND_CHEFS_TOOL],
        },
        "expected": {
            "action": "tool_call",
            "tool_name": "find_chefs",
            "arguments": {
                "chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2,
                "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0,
                "menu": ["水煮鱼"], "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日",
            },
        },
        "generation": {
            "generator": "openai_responses",
            "model": "gpt-5.6-sol",
            "seed": 3001,
            "prompt_sha256": "a" * 64,
            "generated_at": "2026-08-10T00:00:00Z",
        },
        "review": {"status": "machine_validated", "reviewer": None, "notes": []},
        "dpo_targets": [],
    }, ensure_ascii=False)
    sample = parse_raw_sample_line(line)
    assert isinstance(sample, RawBookingSample)
    assert sample.output_kind == sample.expected.action
    assert sample.expected.action == "tool_call"
