"""Phase 03 v0.2 strict raw sample schema — the only fact source for SFT/DPO derivation.

v0.2 changes:
  - RawGenerationMetadata extended with semantic metadata fields
  - DPO heuristics expanded to H1-H8 + H9 (optional)
  - RelativeTimeMetadata, StateTransitionMetadata, ToolFactMetadata added
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from homechef_booking.schemas.decision import FinalDecision, ToolCallDecision
from homechef_booking.schemas.runtime import BookingRuntimeInput

VALID_CONTRACT_ID = "homechef-booking-v1"
VALID_SOURCE = "synthetic"
VALID_DPO_HEURISTICS = {"H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9"}


# ── v0.2 Semantic Metadata Types ─────────────────────────────

class RelativeTimeMetadata(BaseModel):
    """Deterministic relative-time provenance for semantic validation."""
    model_config = ConfigDict(extra="forbid", strict=True)
    expression_type: str
    base_datetime: str
    resolved_service_date: str
    relative_expression: str = ""
    correction_type: str | None = None


class StateTransitionMetadata(BaseModel):
    """State mutation tracking for semantic validation."""
    model_config = ConfigDict(extra="forbid", strict=True)
    changed_fields: list[str] = Field(default_factory=list)
    preserved_fields: list[str] = Field(default_factory=list)
    invalidated_fields: list[str] = Field(default_factory=list)


class ToolFactMetadata(BaseModel):
    """Tool result evidence tracking for semantic validation."""
    model_config = ConfigDict(extra="forbid", strict=True)
    tool_mode: str = ""
    tool_result_status: str = ""
    requested_chef: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    candidate_order: list[str] = Field(default_factory=list)
    evidence_fields: list[str] = Field(default_factory=list)


class RawGenerationMetadata(BaseModel):
    """v0.2: Extended with semantic provenance metadata.

    v0.2.1: Added difficulty field for training curriculum.
    """
    model_config = ConfigDict(extra="forbid", strict=True)
    generator: str = "openai_responses"
    model: str = "gpt-5.6-sol"
    seed: int = 3001
    prompt_sha256: str = Field(min_length=64, max_length=64)
    generated_at: str

    # v0.2 additions
    scenario: str = ""
    capability_tags: list[str] = Field(default_factory=list)
    template_id: str = ""

    relative_time_metadata: RelativeTimeMetadata | None = None
    state_transition_metadata: StateTransitionMetadata | None = None
    tool_fact_metadata: ToolFactMetadata | None = None

    # v0.2.1 additions
    difficulty: Literal["easy", "medium", "hard"] = "medium"


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
    def _validate_consistency(self) -> RawBookingSample:
        if self.output_kind != self.expected.action:
            raise ValueError(
                f"output_kind {self.output_kind} does not match expected.action {self.expected.action}"
            )
        for target in self.dpo_targets:
            if target not in VALID_DPO_HEURISTICS:
                raise ValueError(
                    f"Invalid DPO target {target}. Must be one of {sorted(VALID_DPO_HEURISTICS)}"
                )
        return self


def parse_raw_sample_line(line: str) -> RawBookingSample:
    return RawBookingSample.model_validate(json.loads(line))
