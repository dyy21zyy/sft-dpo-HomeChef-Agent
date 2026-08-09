"""Strict Pydantic v2 BookingSlot, CandidateChef, DecisionState schemas."""

from __future__ import annotations

import datetime
import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.types import StrictBool, StrictFloat, StrictInt, StrictStr

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

AFFIRMATIVE_ALLOWLIST = frozenset({
    "确认",
    "可以",
    "好的",
    "就这样",
    "确认预约",
})


class CandidateChef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chef_id: StrictStr
    chef_name: StrictStr


class BookingSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_date: StrictStr | None = None
    start_time: StrictStr | None = None
    people: StrictInt | None = None
    address: StrictStr | None = None
    cuisine: StrictStr | None = None
    budget_min: StrictFloat | None = None
    budget_max: StrictFloat | None = None
    menu: list[StrictStr] = []
    chef_id: StrictStr | None = None
    chef_name: StrictStr | None = None
    ingredient_purchase: StrictBool | None = None
    dietary_constraints: list[StrictStr] = []
    occasion: StrictStr | None = None
    confirmation: StrictBool | None = None

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


class DecisionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    booking_state: BookingSlot
    chef_query_status: ChefQueryStatus = ChefQueryStatus.not_checked
    candidate_chefs: list[CandidateChef] = []
    awaiting_confirmation: StrictBool = False


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


def is_affirmative(text: str) -> bool:
    """Check if user input is a deterministic affirmative."""
    normalized = text.strip()
    return normalized in AFFIRMATIVE_ALLOWLIST


__all__ = [
    "AFFIRMATIVE_ALLOWLIST",
    "REQUIRED_SLOTS",
    "QUERY_DEPENDENCY_FIELDS",
    "BookingSlot",
    "CandidateChef",
    "ChefQueryStatus",
    "DecisionState",
    "ReplyType",
    "is_affirmative",
    "is_info_complete",
    "missing_required_slots",
]
