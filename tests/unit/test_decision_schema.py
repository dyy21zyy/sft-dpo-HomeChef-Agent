import pytest
from pydantic import ValidationError

from homechef_booking.schemas.decision import ToolCallDecision, parse_decision_obj


def test_tool_call_decision_requires_find_chefs() -> None:
    decision = ToolCallDecision.model_validate({
        "action": "tool_call",
        "tool_name": "find_chefs",
        "arguments": {
            "chef_name": None,
            "service_date": "2026-08-15",
            "start_time": "18:00",
            "people": 6,
            "address": "杨浦",
            "cuisine": "川菜",
            "budget_min": 800.0,
            "budget_max": 1200.0,
            "menu": [],
            "ingredient_purchase": None,
            "dietary_constraints": [],
            "occasion": "家庭聚餐",
        },
    })
    assert decision.tool_name == "find_chefs"


def test_decision_union_rejects_mixed_shape() -> None:
    with pytest.raises(ValidationError):
        parse_decision_obj({"action": "final", "tool_name": "find_chefs", "booking_state": {}})
