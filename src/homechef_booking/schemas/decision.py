"""Strict Pydantic v2 ToolCallDecision, FinalDecision, and Decision union."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from homechef_booking.schemas.booking import (
    BookingSlot,
    CandidateChef,
    ChefQueryStatus,
    ReplyType,
)


class ToolCallDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["tool_call"] = "tool_call"
    tool_name: Literal["find_chefs"] = "find_chefs"
    arguments: dict[str, Any]


class FinalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["final"] = "final"
    booking_state: BookingSlot
    chef_query_status: ChefQueryStatus
    candidate_chefs: list[CandidateChef] = []
    info_complete: bool
    unrelated: bool
    missing_info: list[str] = []
    reply_type: ReplyType
    reply: str | None = None


Decision = Annotated[ToolCallDecision | FinalDecision, Field(discriminator="action")]

_decision_adapter = TypeAdapter(Decision)


def parse_decision_obj(obj: dict[str, Any]) -> ToolCallDecision | FinalDecision:
    """Parse a raw dict into the discriminated Decision union."""
    try:
        return _decision_adapter.validate_python(obj)
    except ValidationError:
        raise


__all__ = [
    "ToolCallDecision",
    "FinalDecision",
    "Decision",
    "parse_decision_obj",
]
