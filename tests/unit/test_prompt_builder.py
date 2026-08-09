"""Task 1: PromptBuilder tests — first model call, tool continuation, and system rules coverage."""

from homechef_booking.prompts import PromptBuilder
from homechef_booking.schemas.runtime import BookingRuntimeInput

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


def _state(address: str | None = "上海市徐汇区") -> dict[str, object]:
    return {
        "booking_state": {
            "service_date": "2026-08-10",
            "start_time": "18:00",
            "people": 2,
            "address": address,
            "cuisine": "川菜",
            "budget_min": 500.0,
            "budget_max": 900.0,
            "menu": ["水煮鱼"],
            "chef_id": None,
            "chef_name": None,
            "ingredient_purchase": True,
            "dietary_constraints": ["不吃花生"],
            "occasion": "生日",
            "confirmation": None,
        },
        "chef_query_status": "not_checked",
        "candidate_chefs": [],
        "awaiting_confirmation": False,
    }


def test_first_model_call_appends_current_user_once():
    runtime_input = BookingRuntimeInput.model_validate({
        "history": [],
        "current_state": _state(),
        "user_input": "明晚六点两个人，川菜，不吃花生",
        "current_time": "2026-08-09 18:00",
        "available_tools": [FIND_CHEFS_TOOL],
    })

    messages = PromptBuilder().build_messages(runtime_input)

    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "明晚六点两个人，川菜，不吃花生"}
    assert [message["role"] for message in messages].count("user") == 1
    assert "current_time" in str(messages[0]["content"])
    assert "current_state" in str(messages[0]["content"])
    assert "find_chefs" in str(messages[0]["content"])


def test_tool_continuation_preserves_native_messages_and_appends_no_user():
    runtime_input = BookingRuntimeInput.model_validate({
        "history": [
            {"role": "user", "content": "帮我找李师傅"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_001", "type": "function", "function": {"name": "find_chefs", "arguments": '{"chef_name":"李师傅","service_date":"2026-08-10","start_time":"18:00","people":2,"address":"上海市徐汇区","cuisine":"川菜","budget_min":500.0,"budget_max":900.0,"menu":["水煮鱼"],"ingredient_purchase":true,"dietary_constraints":["不吃花生"],"occasion":"生日"}'}}]},
            {"role": "tool", "tool_call_id": "call_001", "name": "find_chefs", "content": '{"mode":"specific","status":"available","chef":{"chef_id":"chef_001","chef_name":"李师傅"}}'},
        ],
        "current_state": {
            "booking_state": {"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0, "menu": ["水煮鱼"], "chef_id": None, "chef_name": None, "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日", "confirmation": None},
            "chef_query_status": "available",
            "candidate_chefs": [{"chef_id": "chef_001", "chef_name": "李师傅"}],
            "awaiting_confirmation": False,
        },
        "user_input": None,
        "current_time": "2026-08-09 18:05",
        "available_tools": [FIND_CHEFS_TOOL],
    })

    messages = PromptBuilder().build_messages(runtime_input)

    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert "tool_calls" in messages[2]
    assert messages[3]["role"] == "tool"
    assert messages[3]["tool_call_id"] == "call_001"
    assert len(messages) == 4


def test_system_rules_cover_frozen_booking_behavior_contract():
    runtime_input = BookingRuntimeInput.model_validate({
        "history": [],
        "current_state": _state(),
        "user_input": "明晚六点两个人，川菜，不吃花生",
        "current_time": "2026-08-09 18:00",
        "available_tools": [FIND_CHEFS_TOOL],
    })
    system_content = str(PromptBuilder().build_messages(runtime_input)[0]["content"])
    required_phrases = [
        "today maps to current date, tomorrow maps to +1 day, day-after-tomorrow maps to +2 days",
        "周X maps to the nearest future weekday after current_time",
        "下周X maps to the matching weekday in the next natural week",
        "vague daypart without a concrete clock maps start_time to null",
        "explicit AM/PM or 上午/下午/晚上 clock is normalized only when unambiguous",
        "bare ambiguous clock such as 6点 maps start_time to null",
        "service_date must be a real YYYY-MM-DD calendar date",
        "start_time must be 00:00-23:59",
        "people must be an integer party count",
        "do not infer people from event names or menu quantity",
        "address preserves the user service address and no implicit city completion is allowed",
        "需要买菜 without explicit purchase responsibility maps ingredient_purchase to null",
        "budget mappings are exactly 800到1200 -> 800/1200, 不超过1000 -> null/1000, 最多1000 -> null/1000, 至少1000 -> 1000/null, 预算1000 -> null/1000, 1000左右 -> null/null, 大概1000 -> null/null",
        "dietary_constraints are inherited until explicitly changed",
        "required slots are service_date, start_time, people, address",
        "call find_chefs only after all required slots are non-null",
        "latest valid Tool Result controls chef_query_status",
        "search/matched maps to reply_type present_chef_candidates and preserves Tool candidate order",
        "search/no_match maps to reply_type inform_no_match and candidates []",
        "search/out_of_service_area maps to reply_type inform_out_of_service_area and candidates []",
        "search/error maps to reply_type booking_paused and no retry",
        "specific/available maps to reply_type confirm_specific_chef",
        "specific/unavailable maps to reply_type present_alternatives, preserves alternatives, and has no auto Top-1 semantics",
        "specific/not_found maps to reply_type inform_not_found and alternatives []",
        "specific/out_of_service_area maps to reply_type inform_out_of_service_area and alternatives []",
        "specific/error maps to reply_type booking_paused and no retry",
        "query dependency mutation invalidates prior Tool-derived chef facts",
        "candidate order must match Tool candidate or alternative order",
        "affirmative allowlist is exactly 确认, 可以, 好的, 就这样, 确认预约",
        "mutation dominates confirmation",
        "unrelated true requires reply_type handoff",
        "never claim booking/order creation or completion success",
    ]
    for phrase in required_phrases:
        assert phrase in system_content
