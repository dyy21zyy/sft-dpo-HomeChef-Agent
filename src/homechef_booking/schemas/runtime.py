"""Strict Pydantic v2 BookingRuntimeInput with typed fields."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
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

_FIND_CHEFS_REQUIRED_KEYS = frozenset({
    "chef_name",
    "service_date",
    "start_time",
    "people",
    "address",
    "cuisine",
    "budget_min",
    "budget_max",
    "menu",
    "ingredient_purchase",
    "dietary_constraints",
    "occasion",
})


class ToolFunctionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["find_chefs"]
    description: StrictStr
    parameters: dict

    @model_validator(mode="after")
    def validate_parameters_match_find_chefs_input(self) -> ToolFunctionSpec:
        params = self.parameters
        if not isinstance(params, dict):
            return self
        if params.get("type") != "object":
            raise ValueError(
                "Tool parameters.type must be 'object'"
            )
        if params.get("additionalProperties") is not False:
            raise ValueError(
                "Tool parameters must have additionalProperties: false"
            )
        required = params.get("required")
        if not isinstance(required, list) or sorted(required) != sorted(
            _FIND_CHEFS_REQUIRED_KEYS
        ):
            raise ValueError(
                "Tool parameters.required must be exactly the 12 "
                "FindChefsInput keys"
            )
        properties = params.get("properties")
        if not isinstance(properties, dict):
            raise ValueError("Tool parameters.properties must be a dict")
        prop_keys = set(properties.keys())
        if prop_keys != _FIND_CHEFS_REQUIRED_KEYS:
            extra = prop_keys - _FIND_CHEFS_REQUIRED_KEYS
            missing = _FIND_CHEFS_REQUIRED_KEYS - prop_keys
            msg = (
                "Tool parameters.properties must be exactly the 12 "
                "FindChefsInput keys"
            )
            if extra:
                msg += f", extra: {sorted(extra)}"
            if missing:
                msg += f", missing: {sorted(missing)}"
            raise ValueError(msg)
        return self


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
