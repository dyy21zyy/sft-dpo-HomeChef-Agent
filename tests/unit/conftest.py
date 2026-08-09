import pytest

from homechef_booking.evaluation.sample import EvalCase


def base_input() -> dict[str, object]:
    return {"history": [], "current_state": {"booking_state": {"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0, "menu": ["水煮鱼"], "chef_id": None, "chef_name": None, "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日", "confirmation": None}, "chef_query_status": "matched", "candidate_chefs": [{"chef_id": "chef_001", "chef_name": "李师傅"}], "awaiting_confirmation": False}, "user_input": "明晚六点两个人，川菜，不吃花生", "current_time": "2026-08-09 18:00", "available_tools": [find_chefs_tool_spec()]}


def find_chefs_tool_spec() -> dict[str, object]:
    return {"type": "function", "function": {"name": "find_chefs", "description": "Find chefs", "parameters": {"type": "object", "additionalProperties": False, "required": ["chef_name", "service_date", "start_time", "people", "address", "cuisine", "budget_min", "budget_max", "menu", "ingredient_purchase", "dietary_constraints", "occasion"], "properties": {"chef_name": {"type": ["string", "null"]}, "service_date": {"type": ["string", "null"], "pattern": r"^\d{4}-\d{2}-\d{2}$", "format": "date"}, "start_time": {"type": ["string", "null"], "pattern": r"^([01]\d|2[0-3]):[0-5]\d$"}, "people": {"type": ["integer", "null"]}, "address": {"type": ["string", "null"]}, "cuisine": {"type": ["string", "null"]}, "budget_min": {"type": ["number", "null"]}, "budget_max": {"type": ["number", "null"]}, "menu": {"type": "array", "items": {"type": "string"}}, "ingredient_purchase": {"type": ["boolean", "null"]}, "dietary_constraints": {"type": "array", "items": {"type": "string"}}, "occasion": {"type": ["string", "null"]}}}}}


@pytest.fixture
def valid_tool_call_case() -> EvalCase:
    return EvalCase.model_validate({"id": "case_valid_search_tool_call", "output_kind": "tool_call", "conversation_kind": "single_turn", "input": base_input(), "expected": {"action": "tool_call", "tool_name": "find_chefs", "arguments": {"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0, "menu": ["水煮鱼"], "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日"}}, "assertions": [], "tags": ["valid_search_tool_call"], "reply_expectations": {}})


@pytest.fixture
def valid_final_case() -> EvalCase:
    return EvalCase.model_validate({"id": "case_candidate_selection", "output_kind": "final", "conversation_kind": "multi_turn", "input": base_input(), "expected": {"action": "final", "booking_state": {"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0, "menu": ["水煮鱼"], "chef_id": "chef_001", "chef_name": "李师傅", "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日", "confirmation": None}, "chef_query_status": "matched", "candidate_chefs": [{"chef_id": "chef_001", "chef_name": "李师傅"}], "info_complete": True, "unrelated": False, "missing_info": [], "reply_type": "confirm_specific_chef", "reply": "可以选择李师傅，是否确认？"}, "assertions": [], "tags": ["candidate_selection"], "reply_expectations": {"asks_confirmation_for": "李师傅"}})
