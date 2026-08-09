"""Strict Pydantic v2 FindChefsInput — Task 4 extends result schemas."""

from __future__ import annotations

import datetime
import re

from pydantic import BaseModel, ConfigDict, field_validator

_START_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class FindChefsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chef_name: str | None = None
    service_date: str | None = None
    start_time: str | None = None
    people: int | None = None
    address: str | None = None
    cuisine: str | None = None
    budget_min: float | None = None
    budget_max: float | None = None
    menu: list[str] = []
    ingredient_purchase: bool | None = None
    dietary_constraints: list[str] = []
    occasion: str | None = None

    @field_validator("service_date")
    @classmethod
    def validate_service_date(cls, v: str | None) -> str | None:
        if v is None:
            return None
        try:
            datetime.date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid calendar date: {v}")
        return v

    @field_validator("start_time")
    @classmethod
    def validate_start_time(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not _START_TIME_RE.match(v):
            raise ValueError(f"Invalid start_time: {v}")
        return v


__all__ = ["FindChefsInput"]
