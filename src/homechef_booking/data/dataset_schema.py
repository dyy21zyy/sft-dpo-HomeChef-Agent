"""Phase 03 dataset schemas: SftSample, DpoPair, DatasetManifest, DatasetDataCard."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SftMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: str
    content: str


class SftSample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    raw_id: str
    dataset_version: str
    messages: list[SftMessage]
    prompt_sha256: str = Field(min_length=64, max_length=64)
    completion_sha256: str = Field(min_length=64, max_length=64)
    tags: list[str] = Field(default_factory=list)


class DpoPair(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    raw_id: str
    dataset_version: str
    heuristic: str
    prompt: list[SftMessage]
    chosen: str
    rejected: str
    chosen_sha256: str = Field(min_length=64, max_length=64)
    rejected_sha256: str = Field(min_length=64, max_length=64)
    tags: list[str] = Field(default_factory=list)


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    dataset_version: str
    contract_id: str = "homechef-booking-v1"
    created_at: str
    raw_path: str
    raw_sha256: str
    raw_count: int
    sft_train_path: str | None = None
    sft_train_sha256: str | None = None
    sft_train_count: int = 0
    sft_val_path: str | None = None
    sft_val_sha256: str | None = None
    sft_val_count: int = 0
    dpo_train_path: str | None = None
    dpo_train_sha256: str | None = None
    dpo_train_count: int = 0
    dpo_val_path: str | None = None
    dpo_val_sha256: str | None = None
    dpo_val_count: int = 0
    frozen_eval_overlap: int = 0
    diagnostic_dev_overlap: int = 0
    source: str = "synthetic"
    generator: str = "openai_responses"
    seed: int = 3001


class DatasetDataCard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    phase: str = "03"
    contract_id: str = "homechef-booking-v1"
    dataset_version: str
    created_at: str
    raw_is_fact_source: bool = True
    no_real_user_logs: bool = True
    no_frozen_test_training_use: bool = True
    phase02_base_benchmark_dependency: str = "not_used_for_training"
    known_invalid_phase02_output: str = "0.6B Frozen missing_mock_prediction is invalid for acceptance and not used"
    frozen_eval_overlap: int = 0
    diagnostic_dev_overlap: int = 0
