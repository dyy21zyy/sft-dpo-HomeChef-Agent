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

# Prefixes of natural-language affirmatives (approved in the 25 style anchors,
# e.g. anchor #8 "可以，就订李师傅吧。"). A user input that starts with any of
# these is a deterministic affirmative in real Chinese usage.
AFFIRMATIVE_PREFIXES: tuple[str, ...] = (
    "可以",
    "确认",
    "好的",
    "行",
    "没问题",
    "就订",
    "订",
    "就这样",
    "好，",
)


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
    budget_min: StrictFloat | StrictInt | None = None
    budget_max: StrictFloat | StrictInt | None = None
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
    """Check if user input is a deterministic affirmative.

    Recognizes:
      - an exact AFFIRMATIVE_ALLOWLIST token, OR
      - a natural-language affirmative that starts with an AFFIRMATIVE_PREFIX
        (e.g. "可以，就订李师傅吧", "好的，确认预约", "没问题，就他吧"), OR
      - a conversational hedge that contains a clear affirmative token with NO
        negation (e.g. "现在想的是，确认哈。", "我这边的意思是，可以，就这么订"),
        reflecting real Chinese usage while rejecting genuine negations.
    """
    if not text:
        return False
    normalized = text.strip()

    # Exact allowlist match.
    if normalized in AFFIRMATIVE_ALLOWLIST:
        return True

    # Negation guard: any negation cue → not affirmative.
    if any(c in normalized for c in ("不", "别", "取消", "不要", "不用")):
        return False

    # Prefix match.
    for prefix in AFFIRMATIVE_PREFIXES:
        if normalized.startswith(prefix):
            return True

    # Contains a clear affirmative token (handles conversational hedges).
    strong_affirm_tokens = (
        "确认",
        "没问题",
        "就订",
        "就他",
        "就这个",
        "就这样",
        "可以，就",
        "可以，就这么",
    )
    if any(t in normalized for t in strong_affirm_tokens):
        return True

    return False


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
