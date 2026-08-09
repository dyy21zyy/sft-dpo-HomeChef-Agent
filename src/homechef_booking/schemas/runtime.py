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

# Canonical find_chefs parameters contract — single source of truth
# aligned with contracts/find_chefs_v1.schema.json -> FindChefsInput
_CANONICAL_FIND_CHEFS_PROPERTIES: dict = {
    "chef_name": {"type": ["string", "null"]},
    "service_date": {
        "type": ["string", "null"],
        "pattern": r"^\d{4}-\d{2}-\d{2}$",
        "format": "date",
    },
    "start_time": {
        "type": ["string", "null"],
        "pattern": r"^([01]\d|2[0-3]):[0-5]\d$",
    },
    "people": {"type": ["integer", "null"]},
    "address": {"type": ["string", "null"]},
    "cuisine": {"type": ["string", "null"]},
    "budget_min": {"type": ["number", "null"]},
    "budget_max": {"type": ["number", "null"]},
    "menu": {"type": "array", "items": {"type": "string"}},
    "ingredient_purchase": {"type": ["boolean", "null"]},
    "dietary_constraints": {"type": "array", "items": {"type": "string"}},
    "occasion": {"type": ["string", "null"]},
}


def _normalize_prop_def(prop: object) -> object:
    """Normalize a property definition for semantic comparison.

    Converts regex pattern strings and type lists to sorted tuples so that
    semantically identical definitions compare equal.
    """
    if not isinstance(prop, dict):
        return prop
    result: dict = {}
    for k, v in prop.items():
        if k == "type" and isinstance(v, list):
            result[k] = tuple(sorted(str(t) for t in v))
        else:
            result[k] = v
    return result


def _properties_match(
    actual: dict,
    canonical: dict,
) -> tuple[bool, str]:
    """Compare actual ToolSpec properties against canonical contract.

    Returns (match: bool, error: str).
    """
    actual_keys = set(actual.keys())
    canonical_keys = set(canonical.keys())

    if actual_keys != canonical_keys:
        extra = actual_keys - canonical_keys
        missing = canonical_keys - actual_keys
        msg = "properties keys mismatch"
        if extra:
            msg += f", extra: {sorted(extra)}"
        if missing:
            msg += f", missing: {sorted(missing)}"
        return False, msg

    for key in canonical_keys:
        actual_norm = _normalize_prop_def(actual[key])
        canon_norm = _normalize_prop_def(canonical[key])
        if actual_norm != canon_norm:
            return False, f"property '{key}' definition mismatch"

    return True, ""


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
            raise ValueError("Tool parameters.type must be 'object'")
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

        match_ok, err = _properties_match(properties, _CANONICAL_FIND_CHEFS_PROPERTIES)
        if not match_ok:
            raise ValueError(f"Tool parameters.properties: {err}")

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
