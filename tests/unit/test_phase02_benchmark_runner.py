"""Task 5 Phase 02: Benchmark runner tests."""

import json
from pathlib import Path

from homechef_booking.evaluation.benchmark_runner import (
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark,
)


def test_benchmark_config_loads_from_yaml(tmp_path: Path):
    config_yaml = tmp_path / "benchmark.yaml"
    config_yaml.write_text("\n".join([
        "run_id: test_benchmark",
        "cases_path: data/eval/frozen_test.jsonl",
        "backend_config_path: configs/inference/mock.yaml",
        "manifest_path: data/eval/frozen_test.manifest.json",
        "output_dir: reports/generated",
        "model_id: Qwen/Qwen3-0.6B-Base",
    ]), encoding="utf-8")
    config = BenchmarkConfig.load_yaml(config_yaml)
    assert config.run_id == "test_benchmark"
    assert config.model_id == "Qwen/Qwen3-0.6B-Base"


def test_benchmark_result_model_has_all_fields():
    result = BenchmarkResult(
        run_id="test_run",
        model_id="Qwen/Qwen3-0.6B-Base",
        suite_id="test_suite",
        total_cases=120,
        protocol_pass_rate=0.95,
        effective_pass_rate=0.80,
        mean_task_correctness=0.85,
        mean_structured_score=0.90,
        mean_reply_score=0.75,
        critical_error_rate=0.10,
        dimension_pass_rates={"protocol": 0.95, "task_correctness": 0.80},
        assertion_pass_rates={"tool_fact_grounded": 0.95},
        critical_error_distribution={"chef_fabrication": 5},
        metrics={"tool_arguments_accuracy": 0.95},
        latency_metrics={"mean_latency_ms": 500.0},
        hardware_info={"device": "cpu", "torch_dtype": "float32"},
    )
    assert result.run_id == "test_run"
    assert result.total_cases == 120
    assert result.metrics["tool_arguments_accuracy"] == 0.95


def test_run_benchmark_mock_produces_complete_result(tmp_path: Path):
    config = BenchmarkConfig(
        run_id="test_bench",
        cases_path=Path("tests/fixtures/evaluation/phase01_mock_cases.jsonl"),
        backend_config_path=Path("configs/inference/mock.yaml"),
        predictions_path=Path("tests/fixtures/evaluation/phase01_mock_predictions.json"),
        output_dir=tmp_path,
        model_id="Qwen/Qwen3-0.6B-Base",
        suite_id="phase01_mock",
        manifest_path=None,
    )
    result = run_benchmark(config)
    assert result.total_cases == 19
    assert 0 <= result.effective_pass_rate <= 1
    assert result.model_id == "Qwen/Qwen3-0.6B-Base"
    assert result.suite_id == "phase01_mock"
    assert result.latency_metrics is not None
    assert result.hardware_info is not None
    scorecard_path = tmp_path / "test_bench_scorecard.json"
    assert scorecard_path.exists()
    results_path = tmp_path / "test_bench_case_results.json"
    assert results_path.exists()
