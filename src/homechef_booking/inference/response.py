from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class GenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    case_id: str
    backend_name: str
    raw_text: str | None = None
    finish_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    tokens_per_second: float | None = None
    rss_mb: float | None = None
    model_size_bytes: int | None = None
