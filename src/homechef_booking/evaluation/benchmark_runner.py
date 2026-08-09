"""Phase 02 benchmark runner for frozen eval with real or mock backends."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone

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
    max_model_length: int = 2048
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    @classmethod
    def load_yaml(cls, path: Path) -> "BenchmarkConfig":
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
            max_model_length=data.get("max_model_length", 2048),
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
        )


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
    latency_metrics: dict[str, float] | None = None
    hardware_info: dict[str, str] | None = None
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


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
    latency_metrics = {
        "total_wall_seconds": round(elapsed, 3),
        "mean_wall_seconds_per_case": round(elapsed / max(scorecard.metrics.total_cases, 1), 3),
    }
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
