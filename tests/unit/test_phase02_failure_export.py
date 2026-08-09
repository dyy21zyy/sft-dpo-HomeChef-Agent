"""Task 6 Phase 02: Failure analysis export tests."""

import json
from pathlib import Path

from homechef_booking.evaluation.benchmark_runner import BenchmarkConfig, run_benchmark
from homechef_booking.evaluation.failure_export import export_failure_report


def test_failure_export_creates_structured_report(tmp_path: Path):
    config = BenchmarkConfig(
        run_id="test_failure",
        cases_path=Path("tests/fixtures/evaluation/phase01_mock_cases.jsonl"),
        backend_config_path=Path("configs/inference/mock.yaml"),
        predictions_path=Path("tests/fixtures/evaluation/phase01_mock_predictions.json"),
        output_dir=tmp_path,
        model_id="Qwen/Qwen3-0.6B-Base",
        suite_id="phase01_mock",
        manifest_path=None,
    )
    result = run_benchmark(config)
    case_results_path = tmp_path / "test_failure_case_results.json"
    report_path = export_failure_report(case_results_path, tmp_path / "failure_report.json")
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "suite_id" in report
    assert "failed_cases" in report
    assert "critical_error_cases" in report
    assert isinstance(report["failed_cases"], list)
    assert isinstance(report["critical_error_cases"], list)
    assert "summary" in report
    assert "total_cases" in report["summary"]
    assert "total_failed" in report["summary"]
