"""Phase 02 data card generation for benchmark reports."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from homechef_booking.evaluation.benchmark_runner import BenchmarkConfig, BenchmarkResult


def generate_data_card(result: BenchmarkResult, config: BenchmarkConfig, output_path: Path) -> Path:
    latency = result.latency_metrics
    card = {
        "benchmark_name": "HomeChef-Booking-M0",
        "phase": "02",
        "contract_version": "homechef-booking-v1",
        "model_id": result.model_id,
        "suite_id": result.suite_id,
        "run_id": result.run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "frozen_suite": config.manifest_path is not None,
        "runtime": config.runtime or "unknown",
        "model_format": config.model_format or "unknown",
        "quantization": config.quantization or "unknown",
        "gpu_layers": config.gpu_layers,
        "device": config.device,
        "no_sft_dpo_training": True,
        "no_training": True,
        "hardware": result.hardware_info or {},
        "latency": {
            "mean_latency_ms": latency.mean_latency_ms if latency else None,
            "median_latency_ms": latency.median_latency_ms if latency else None,
            "p95_latency_ms": latency.p95_latency_ms if latency else None,
            "mean_ttft_ms": latency.mean_ttft_ms if latency else None,
            "median_ttft_ms": latency.median_ttft_ms if latency else None,
            "p95_ttft_ms": latency.p95_ttft_ms if latency else None,
            "mean_tokens_per_second": latency.mean_tokens_per_second if latency else None,
            "median_tokens_per_second": latency.median_tokens_per_second if latency else None,
            "performance_sample_count": latency.performance_sample_count if latency else 0,
            "failed_inference_cases": latency.failed_inference_cases if latency else 0,
            "timeout_cases": latency.timeout_cases if latency else 0,
            "total_wall_seconds": latency.total_wall_seconds if latency else 0,
            "mean_wall_seconds_per_case": latency.mean_wall_seconds_per_case if latency else 0,
        },
        "results": {
            "total_cases": result.total_cases,
            "protocol_pass_rate": result.protocol_pass_rate,
            "effective_pass_rate": result.effective_pass_rate,
            "mean_task_correctness": result.mean_task_correctness,
            "mean_structured_score": result.mean_structured_score,
            "mean_reply_score": result.mean_reply_score,
            "critical_error_rate": result.critical_error_rate,
        },
        "dimension_pass_rates": result.dimension_pass_rates,
        "assertion_pass_rates": result.assertion_pass_rates,
        "critical_error_distribution": result.critical_error_distribution,
        "metrics": result.metrics,
        "notes": config.notes,
        "tags": config.tags,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
