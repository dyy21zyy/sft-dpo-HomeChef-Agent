"""Phase 03 strict raw sample schema — the only fact source for SFT/DPO derivation."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from homechef_booking.schemas.decision import FinalDecision, ToolCallDecision
from homechef_booking.schemas.runtime import BookingRuntimeInput

VALID_CONTRACT_ID = "homechef-booking-v1"
VALID_SOURCE = "synthetic"
VALID_DPO_HEURISTICS = {"H1", "H2", "H3", "H4", "H5", "H6", "H7"}


class RawGenerationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    generator: str = "openai_responses"
    model: str = "gpt-5.6-sol"
    seed: int = 3001
    prompt_sha256: str = Field(min_length=64, max_length=64)
    generated_at: str


class RawReviewMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["machine_validated", "human_reviewed", "rejected"] = "machine_validated"
    reviewer: str | None = None
    notes: list[str] = Field(default_factory=list)


class RawBookingSample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    dataset_version: str
    contract_id: Literal["homechef-booking-v1"] = "homechef-booking-v1"
    source: Literal["synthetic"] = "synthetic"
    scenario: str
    output_kind: Literal["tool_call", "final"]
    conversation_kind: Literal["single_turn", "multi_turn"]
    tags: list[str] = Field(default_factory=list)
    input: BookingRuntimeInput
    expected: ToolCallDecision | FinalDecision
    generation: RawGenerationMetadata
    review: RawReviewMetadata = Field(default_factory=RawReviewMetadata)
    dpo_targets: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_consistency(self) -> "RawBookingSample":
        if self.output_kind != self.expected.action:
            raise ValueError(f"output_kind {self.output_kind} does not match expected.action {self.expected.action}")
        for target in self.dpo_targets:
            if target not in VALID_DPO_HEURISTICS:
                raise ValueError(f"Invalid DPO target {target}. Must be one of {sorted(VALID_DPO_HEURISTICS)}")
        return self


def parse_raw_sample_line(line: str) -> RawBookingSample:
    return RawBookingSample.model_validate(json.loads(line))
