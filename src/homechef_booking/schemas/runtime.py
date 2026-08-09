"""Strict Pydantic v2 BookingRuntimeInput with typed fields."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.types import StrictStr

from homechef_booking.schemas.booking import DecisionState
from homechef_booking.schemas.history import (
    AssistantTextMessage,
    AssistantToolCallMessage,
    ToolMessage,
    UserMessage,
)

_CURRENT_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} ([01]\d|2[0-3]):[0-5]\d$")

HistoryMessage = (
    UserMessage | AssistantTextMessage | AssistantToolCallMessage | ToolMessage
)


class ToolFunctionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["find_chefs"]
    description: StrictStr
    parameters: dict


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["function"]
    function: ToolFunctionSpec


class BookingRuntimeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history: list[HistoryMessage] = []
    current_state: DecisionState
    user_input: StrictStr | None = None
    current_time: StrictStr
    available_tools: list[ToolSpec] = []

    @field_validator("current_time")
    @classmethod
    def validate_current_time(cls, v: StrictStr) -> StrictStr:
        if not _CURRENT_TIME_RE.match(v):
            raise ValueError(f"Invalid current_time: {v}")
        return v


__all__ = [
    "BookingRuntimeInput",
    "HistoryMessage",
    "ToolFunctionSpec",
    "ToolSpec",
]
