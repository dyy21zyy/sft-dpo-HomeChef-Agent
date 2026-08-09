"""Strict Pydantic v2 ToolCallDecision, FinalDecision, and Decision union."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic.types import StrictBool, StrictStr

from homechef_booking.schemas.booking import (
    BookingSlot,
    CandidateChef,
    ChefQueryStatus,
    ReplyType,
)
from homechef_booking.schemas.tools import FindChefsInput


class ToolCallDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["tool_call"] = "tool_call"
    tool_name: Literal["find_chefs"] = "find_chefs"
    arguments: FindChefsInput


class FinalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["final"] = "final"
    booking_state: BookingSlot
    chef_query_status: ChefQueryStatus
    candidate_chefs: list[CandidateChef] = []
    info_complete: StrictBool
    unrelated: StrictBool
    missing_info: list[StrictStr] = []
    reply_type: ReplyType
    reply: StrictStr | None = None


def parse_decision_obj(obj: dict) -> ToolCallDecision | FinalDecision:
    """Parse a raw dict into the discriminated Decision union.

    Returns a predictable validation error for invalid actions
    without crashing or fabricating internal Pydantic exceptions.
    """
    action = obj.get("action")
    if action == "tool_call":
        return ToolCallDecision.model_validate(obj)
    if action == "final":
        return FinalDecision.model_validate(obj)
    raise ValidationError.from_exception_data(
        title="Decision",
        line_errors=[{
            "type": "literal_error",
            "loc": ("action",),
            "msg": (
                "Input should be 'tool_call' or 'final'"
                f", got {repr(action)}"
            ),
            "input": action,
            "ctx": {"expected": "'tool_call' or 'final'"},
        }],
    )


__all__ = [
    "FinalDecision",
    "ToolCallDecision",
    "parse_decision_obj",
]
