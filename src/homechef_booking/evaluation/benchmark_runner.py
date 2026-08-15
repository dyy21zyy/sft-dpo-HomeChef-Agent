"""Phase 02 benchmark runner for frozen eval with real or mock backends."""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from homechef_booking.evaluation.runner import EvalConfig, run_evaluation
from homechef_booking.evaluation.suite_manifest import validate_suite_manifest


@dataclass
class BenchmarkConfig:
    run_id: str
    cases_path: Path
    backend_config_path: Path
    output_dir: Path
    model_id: str
    suite_id: str = ""
    manifest_path: Path | None = None
    predictions_path: Path | None = None
    device: str = "auto"
    torch_dtype: str = "float32"
    runtime: str = ""
    model_format: str = ""
    quantization: str = ""
    gpu_layers: int = 0
    max_model_length: int = 2048
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    @classmethod
    def load_yaml(cls, path: Path) -> BenchmarkConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            run_id=data["run_id"],
            cases_path=Path(data["cases_path"]),
            backend_config_path=Path(data["backend_config_path"]),
            output_dir=Path(data.get("output_dir", "reports/generated")),
            model_id=data.get("model_id", "unknown"),
            suite_id=data.get("suite_id", ""),
            manifest_path=Path(data["manifest_path"]) if data.get("manifest_path") else None,
            predictions_path=Path(data["predictions_path"]) if data.get("predictions_path") else None,
            device=data.get("device", "auto"),
            torch_dtype=data.get("torch_dtype", "float32"),
            runtime=data.get("runtime", ""),
            model_format=data.get("model_format", ""),
            quantization=data.get("quantization", ""),
            gpu_layers=data.get("gpu_layers", 0),
            max_model_length=data.get("max_model_length", 2048),
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
        )


class LatencyMetrics(BaseModel):
    """Real model inference latency metrics (not E2E wall time)."""
    model_config = ConfigDict(extra="forbid", strict=True)
    mean_latency_ms: float | None = None
    median_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    mean_ttft_ms: float | None = None
    median_ttft_ms: float | None = None
    p95_ttft_ms: float | None = None
    mean_tokens_per_second: float | None = None
    median_tokens_per_second: float | None = None
    performance_sample_count: int = 0
    successful_inference_cases: int = 0
    failed_inference_cases: int = 0
    timeout_cases: int = 0
    total_wall_seconds: float = 0.0
    mean_wall_seconds_per_case: float = 0.0


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    run_id: str
    model_id: str
    suite_id: str
    total_cases: int
    protocol_pass_rate: float
    effective_pass_rate: float
    mean_task_correctness: float
    mean_structured_score: float
    mean_reply_score: float
    critical_error_rate: float
    dimension_pass_rates: dict[str, float] = Field(default_factory=dict)
    assertion_pass_rates: dict[str, float] = Field(default_factory=dict)
    critical_error_distribution: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, float | str] = Field(default_factory=dict)
    latency_metrics: LatencyMetrics | None = None
    hardware_info: dict[str, str] | None = None
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def _compute_percentile(sorted_values: list[float], percentile: float) -> float | None:
    """Compute percentile using linear interpolation."""
    if not sorted_values:
        return None
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    k = (percentile / 100.0) * (n - 1)
    f = int(k)
    c = k - f
    if f + 1 < n:
        return sorted_values[f] + c * (sorted_values[f + 1] - sorted_values[f])
    return sorted_values[f]


def _compute_latency_metrics(
    case_results: list[dict],
    total_wall_seconds: float,
    total_cases: int,
) -> LatencyMetrics:
    """Extract real model inference latency metrics from per-case GenerationResult data."""
    latencies: list[float] = []
    ttfts: list[float] = []
    throughputs: list[float] = []
    failed = 0
    timeouts = 0
    successful = 0

    for cr in case_results:
        gen = cr.get("generation_result")
        if gen is None:
            failed += 1
            continue
        error = gen.get("error_type")
        if error:
            failed += 1
            if error == "timeout":
                timeouts += 1
            continue
        successful += 1
        lat = gen.get("latency_ms")
        if lat is not None and lat > 0:
            latencies.append(lat)
        ttft = gen.get("ttft_ms")
        if ttft is not None and ttft > 0:
            ttfts.append(ttft)
        tps = gen.get("tokens_per_second")
        if tps is not None and tps > 0:
            throughputs.append(tps)

    perf_count = len(latencies)
    latencies_sorted = sorted(latencies)
    ttfts_sorted = sorted(ttfts)

    return LatencyMetrics(
        mean_latency_ms=round(statistics.mean(latencies), 2) if latencies else None,
        median_latency_ms=round(statistics.median(latencies), 2) if latencies else None,
        p95_latency_ms=round(_compute_percentile(latencies_sorted, 95), 2) if latencies_sorted else None,
        mean_ttft_ms=round(statistics.mean(ttfts), 2) if ttfts else None,
        median_ttft_ms=round(statistics.median(ttfts), 2) if ttfts else None,
        p95_ttft_ms=round(_compute_percentile(ttfts_sorted, 95), 2) if ttfts_sorted else None,
        mean_tokens_per_second=round(statistics.mean(throughputs), 2) if throughputs else None,
        median_tokens_per_second=round(statistics.median(throughputs), 2) if throughputs else None,
        performance_sample_count=perf_count,
        successful_inference_cases=successful,
        failed_inference_cases=failed,
        timeout_cases=timeouts,
        total_wall_seconds=round(total_wall_seconds, 3),
        mean_wall_seconds_per_case=round(total_wall_seconds / max(total_cases, 1), 3),
    )


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    if config.manifest_path:
        validate_suite_manifest(config.manifest_path, root=Path("."))
    eval_config = EvalConfig(
        cases_path=config.cases_path,
        backend_config_path=config.backend_config_path,
        predictions_path=config.predictions_path,
        scorecard_path=config.output_dir / f"{config.run_id}_scorecard.json",
        case_results_path=config.output_dir / f"{config.run_id}_case_results.json",
    )
    start = time.perf_counter()
    eval_result = run_evaluation(eval_config)
    elapsed = time.perf_counter() - start
    scorecard = eval_result.scorecard

    hardware_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "device": config.device,
        "torch_dtype": config.torch_dtype,
    }
    if config.runtime:
        hardware_info["runtime"] = config.runtime
    if config.model_format:
        hardware_info["model_format"] = config.model_format
    if config.quantization:
        hardware_info["quantization"] = config.quantization
    hardware_info["gpu_layers"] = str(config.gpu_layers)

    # Build per-case data for latency computation
    case_data = []
    for case_result, gen_result in zip(eval_result.case_results, eval_result.generation_results, strict=True):
        cr_dict = case_result.model_dump(mode="json")
        cr_dict["generation_result"] = gen_result.model_dump(mode="json") if gen_result else None
        case_data.append(cr_dict)

    latency_metrics = _compute_latency_metrics(case_data, elapsed, scorecard.metrics.total_cases)

    result = BenchmarkResult(
        run_id=config.run_id,
        model_id=config.model_id,
        suite_id=config.suite_id,
        total_cases=scorecard.metrics.total_cases,
        protocol_pass_rate=scorecard.metrics.protocol_pass_rate,
        effective_pass_rate=scorecard.metrics.effective_pass_rate,
        mean_task_correctness=scorecard.metrics.mean_task_correctness,
        mean_structured_score=scorecard.metrics.mean_structured_score,
        mean_reply_score=scorecard.metrics.mean_reply_score,
        critical_error_rate=scorecard.metrics.critical_error_rate,
        dimension_pass_rates=scorecard.dimension_pass_rates,
        assertion_pass_rates=scorecard.assertion_pass_rates,
        critical_error_distribution=scorecard.critical_error_distribution,
        metrics={k: v for k, v in scorecard.metrics.metrics.items()},
        latency_metrics=latency_metrics,
        hardware_info=hardware_info,
        notes=config.notes,
        tags=config.tags,
    )
    result_path = config.output_dir / f"{config.run_id}_benchmark.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result.model_dump(mode="json", exclude_none=False), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return result
