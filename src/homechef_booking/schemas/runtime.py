"""Strict Pydantic v2 BookingRuntimeInput."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BookingRuntimeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history: list[dict] = []
    current_state: dict
    user_input: str | None = None
    current_time: str
    available_tools: list[dict] = []


__all__ = ["BookingRuntimeInput"]
