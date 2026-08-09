from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DimensionScore(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    score: float | None
    passed: bool | None
    details: dict[str, object] = Field(default_factory=dict)


class AssertionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    passed: bool
    details: dict[str, object] = Field(default_factory=dict)


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    protocol_pass: bool
    structured_score: float
    reply_score: float
    task_correctness: float
    critical_error: bool
    critical_error_tags: list[str] = Field(default_factory=list)
    effective_pass: bool
    dimensions: list[DimensionScore] = Field(default_factory=list)
    assertions: list[AssertionResult] = Field(default_factory=list)
    metric_outcomes: dict[str, bool | str] = Field(default_factory=dict)
