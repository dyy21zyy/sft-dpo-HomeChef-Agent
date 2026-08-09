from homechef_booking.schemas.history import parse_history_messages, validate_history_sequence


def test_tool_result_requires_matching_call_id() -> None:
    messages = parse_history_messages([
        {"role": "tool", "tool_call_id": "call_missing", "name": "find_chefs", "content": "{}"}
    ])
    errors = validate_history_sequence(messages, user_input=None)
    assert "tool_call_id has no pending call" in "\n".join(errors)


def test_tool_call_arguments_are_json_string() -> None:
    messages = parse_history_messages([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {"name": "find_chefs", "arguments": "{\"chef_name\":null,\"service_date\":\"2026-08-15\",\"start_time\":\"18:00\",\"people\":6,\"address\":\"杨浦\",\"cuisine\":\"川菜\",\"budget_min\":800.0,\"budget_max\":1200.0,\"menu\":[],\"ingredient_purchase\":null,\"dietary_constraints\":[],\"occasion\":\"家庭聚餐\"}"},  # noqa: E501
            }],
        },
        {"role": "tool", "tool_call_id": "call_001", "name": "find_chefs", "content": "{\"mode\":\"search\",\"status\":\"matched\",\"candidates\":[{\"chef_id\":\"C003\",\"chef_name\":\"张伟\"}]}"},  # noqa: E501
    ])
    assert validate_history_sequence(messages, user_input=None) == []
