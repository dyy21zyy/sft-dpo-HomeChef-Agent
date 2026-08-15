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
    throughput_source: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    server_predicted_tokens: int | None = None
    server_predicted_ms: float | None = None
    server_tokens_per_second: float | None = None
    rss_mb: float | None = None
    model_size_bytes: int | None = None
    # ── Phase04 provenance metadata ─────────────────────────────
    # Enables a report to prove WHICH trained model was evaluated
    # (e.g. 1.7B-SFT) rather than the Base model. adapter provenance must not
    # live only in backend in-memory config.
    base_model_id: str | None = None
    adapter_name_or_path: str | None = None
    training_stage: str | None = None   # "sft" | "dpo" | None
    model_size: str | None = None       # "1.7B" | "4B"
    use_structured_output: bool | None = None
    pref_beta: float | None = None      # None for SFT; 0.1/0.3 for DPO
