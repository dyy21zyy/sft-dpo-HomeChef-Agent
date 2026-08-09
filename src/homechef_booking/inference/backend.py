from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from homechef_booking.inference.response import GenerationResult
from homechef_booking.prompts import Message


class GenerationParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout_seconds: float | None = None


class Backend(Protocol):
    name: str

    def generate(self, messages: list[Message], params: GenerationParams, case_id: str | None = None) -> GenerationResult:
        raise NotImplementedError
