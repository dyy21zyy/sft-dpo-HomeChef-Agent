"""Strict Pydantic v2 BookingSlot, CandidateChef, DecisionState schemas."""

from __future__ import annotations

import datetime
import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.types import StrictBool, StrictInt, StrictStr

_START_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class ChefQueryStatus(StrEnum):
    not_checked = "not_checked"
    matched = "matched"
    available = "available"
    unavailable = "unavailable"
    not_found = "not_found"
    no_match = "no_match"
    out_of_service_area = "out_of_service_area"
    error = "error"


class ReplyType(StrEnum):
    handoff = "handoff"
    ask_service_date = "ask_service_date"
    ask_start_time = "ask_start_time"
    ask_people = "ask_people"
    ask_address = "ask_address"
    ask_multiple_required_fields = "ask_multiple_required_fields"
    present_chef_candidates = "present_chef_candidates"
    confirm_specific_chef = "confirm_specific_chef"
    present_alternatives = "present_alternatives"
    inform_not_found = "inform_not_found"
    inform_no_match = "inform_no_match"
    inform_out_of_service_area = "inform_out_of_service_area"
    booking_authorized = "booking_authorized"
    booking_paused = "booking_paused"
    acknowledge_result = "acknowledge_result"


REQUIRED_SLOTS = ["service_date", "start_time", "people", "address"]

QUERY_DEPENDENCY_FIELDS = [
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
    "chef_name",
]


class CandidateChef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chef_id: StrictStr
    chef_name: StrictStr


class BookingSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_date: str | None = None
    start_time: str | None = None
    people: StrictInt | None = None
    address: str | None = None
    cuisine: str | None = None
    budget_min: float | None = None
    budget_max: float | None = None
    menu: list[str] = []
    chef_id: str | None = None
    chef_name: str | None = None
    ingredient_purchase: StrictBool | None = None
    dietary_constraints: list[str] = []
    occasion: str | None = None
    confirmation: bool | None = None

    @field_validator("service_date")
    @classmethod
    def validate_service_date(cls, v: str | None) -> str | None:
        if v is None:
            return None
        try:
            datetime.date.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"Invalid calendar date: {v}") from exc
        return v

    @field_validator("start_time")
    @classmethod
    def validate_start_time(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not _START_TIME_RE.match(v):
            raise ValueError(f"Invalid start_time: {v}")
        return v


class DecisionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    booking_state: BookingSlot
    chef_query_status: ChefQueryStatus = ChefQueryStatus.not_checked
    candidate_chefs: list[CandidateChef] = []
    awaiting_confirmation: bool = False


def missing_required_slots(slot: BookingSlot) -> list[str]:
    """Return missing required slots in canonical contract order."""
    result: list[str] = []
    if slot.service_date is None:
        result.append("service_date")
    if slot.start_time is None:
        result.append("start_time")
    if slot.people is None:
        result.append("people")
    if slot.address is None:
        result.append("address")
    return result


def is_info_complete(slot: BookingSlot) -> bool:
    """Check if all required slots are filled."""
    return len(missing_required_slots(slot)) == 0


__all__ = [
    "BookingSlot",
    "CandidateChef",
    "ChefQueryStatus",
    "DecisionState",
    "ReplyType",
    "REQUIRED_SLOTS",
    "QUERY_DEPENDENCY_FIELDS",
    "is_info_complete",
    "missing_required_slots",
]
