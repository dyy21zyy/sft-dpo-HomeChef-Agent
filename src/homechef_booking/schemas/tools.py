"""Strict Pydantic v2 find_chefs Tool input and result schemas."""

from __future__ import annotations

import datetime
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic.types import StrictBool, StrictFloat, StrictInt, StrictStr

_START_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

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


class CandidateChef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chef_id: StrictStr
    chef_name: StrictStr


class FindChefsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chef_name: StrictStr | None = None
    service_date: StrictStr | None = None
    start_time: StrictStr | None = None
    people: StrictInt | None = None
    address: StrictStr | None = None
    cuisine: StrictStr | None = None
    budget_min: StrictFloat | None = None
    budget_max: StrictFloat | None = None
    menu: list[StrictStr] = []
    ingredient_purchase: StrictBool | None = None
    dietary_constraints: list[StrictStr] = []
    occasion: StrictStr | None = None

    @model_validator(mode="before")
    @classmethod
    def require_all_12_keys(cls, data: object) -> object:
        if isinstance(data, dict):
            missing = _FIND_CHEFS_REQUIRED_KEYS - set(data.keys())
            if missing:
                raise ValueError(
                    f"FindChefsInput requires all 12 keys, missing: "
                    f"{sorted(missing)}"
                )
        return data

    @field_validator("service_date")
    @classmethod
    def validate_service_date(cls, v: StrictStr | None) -> StrictStr | None:
        if v is None:
            return None
        try:
            datetime.date.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"Invalid calendar date: {v}") from exc
        return v

    @field_validator("start_time")
    @classmethod
    def validate_start_time(cls, v: StrictStr | None) -> StrictStr | None:
        if v is None:
            return None
        if not _START_TIME_RE.match(v):
            raise ValueError(f"Invalid start_time: {v}")
        return v


class SearchMatchedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["search"]
    status: Literal["matched"]
    candidates: list[CandidateChef]


class SearchNoMatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["search"]
    status: Literal["no_match"]


class SearchOutOfServiceAreaResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["search"]
    status: Literal["out_of_service_area"]


class SearchErrorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["search"]
    status: Literal["error"]


class SpecificAvailableResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["specific"]
    status: Literal["available"]


class SpecificUnavailableResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["specific"]
    status: Literal["unavailable"]
    alternatives: list[CandidateChef]


class SpecificNotFoundResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["specific"]
    status: Literal["not_found"]


class SpecificOutOfServiceAreaResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["specific"]
    status: Literal["out_of_service_area"]


class SpecificErrorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["specific"]
    status: Literal["error"]


def parse_find_chefs_result(obj: dict) -> BaseModel:
    """Parse a raw dict into the correct FindChefsResult variant.

    Uses (mode, status) pair to select the exact model, since status alone
    is ambiguous between search and specific modes.
    """
    mode = obj.get("mode")
    status = obj.get("status")
    if mode == "search":
        if status == "matched":
            return SearchMatchedResult.model_validate(obj)
        if status == "no_match":
            return SearchNoMatchResult.model_validate(obj)
        if status == "out_of_service_area":
            return SearchOutOfServiceAreaResult.model_validate(obj)
        if status == "error":
            return SearchErrorResult.model_validate(obj)
    elif mode == "specific":
        if status == "available":
            return SpecificAvailableResult.model_validate(obj)
        if status == "unavailable":
            return SpecificUnavailableResult.model_validate(obj)
        if status == "not_found":
            return SpecificNotFoundResult.model_validate(obj)
        if status == "out_of_service_area":
            return SpecificOutOfServiceAreaResult.model_validate(obj)
        if status == "error":
            return SpecificErrorResult.model_validate(obj)
    raise ValueError(f"Invalid find_chefs result: mode={mode}, status={status}")


__all__ = [
    "CandidateChef",
    "FindChefsInput",
    "SearchErrorResult",
    "SearchMatchedResult",
    "SearchNoMatchResult",
    "SearchOutOfServiceAreaResult",
    "SpecificAvailableResult",
    "SpecificErrorResult",
    "SpecificNotFoundResult",
    "SpecificOutOfServiceAreaResult",
    "SpecificUnavailableResult",
    "parse_find_chefs_result",
]
