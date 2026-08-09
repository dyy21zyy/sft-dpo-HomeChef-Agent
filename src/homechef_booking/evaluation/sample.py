from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from homechef_booking.schemas.decision import FinalDecision, ToolCallDecision
from homechef_booking.schemas.runtime import BookingRuntimeInput

Decision = ToolCallDecision | FinalDecision


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    output_kind: Literal["tool_call", "final"]
    conversation_kind: Literal["single_turn", "multi_turn"]
    input: BookingRuntimeInput
    expected: Decision
    assertions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    reply_expectations: dict[str, object] = Field(default_factory=dict)
    chain_id: str | None = None
    step: int | None = None


def load_eval_cases(path: Path, known_assertions: set[str] | None = None) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        case = EvalCase.model_validate(json.loads(line))
        if case.id in seen:
            raise ValueError(f"Duplicate eval case id {case.id} at line {line_number}")
        if case.output_kind != case.expected.action:
            raise ValueError(f"output_kind {case.output_kind} does not match expected.action {case.expected.action}")
        if known_assertions is not None:
            unknown = [name for name in case.assertions if name not in known_assertions]
            if unknown:
                raise ValueError(f"Unknown assertion names: {', '.join(unknown)}")
        if case.input.user_input is not None and case.input.history and getattr(case.input.history[-1], "role", None) == "user" and getattr(case.input.history[-1], "content", None) == case.input.user_input:
            raise ValueError(f"Eval case {case.id} duplicates current user in history and user_input")
        if case.output_kind == "tool_call" and not case.input.available_tools:
            raise ValueError(f"Eval case {case.id} requires complete find_chefs ToolSpec in available_tools")
        seen.add(case.id)
        cases.append(case)
    return cases
