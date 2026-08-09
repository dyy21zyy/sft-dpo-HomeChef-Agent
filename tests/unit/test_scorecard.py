"""Task 8: Scorecard and runner tests."""

import json
from pathlib import Path

from homechef_booking.evaluation.runner import EvalConfig, run_evaluation
from homechef_booking.evaluation.scorecard import aggregate_scorecard


PREDICTIONS_PATH = Path("tests/fixtures/evaluation/phase01_mock_predictions.json")


def test_eval_runner_handles_19_mock_cases(tmp_path: Path):
    config = EvalConfig(
        cases_path=Path("tests/fixtures/evaluation/phase01_mock_cases.jsonl"),
        backend_config_path=Path("configs/inference/mock.yaml"),
        predictions_path=PREDICTIONS_PATH,
        scorecard_path=tmp_path / "scorecard.json",
        case_results_path=tmp_path / "case_results.json",
    )
    result = run_evaluation(config)
    assert len(result.case_results) == 19
    assert result.scorecard.metrics.total_cases == 19
    assert result.scorecard.metrics.effective_pass_rate > 0
    assert result.scorecard.phase == "01"
    assert "tool_arguments_accuracy" in result.scorecard.metrics.metrics
    assert result.scorecard.metrics.metrics["tool_arguments_accuracy"] != "not_available"


def test_scorecard_aggregation_produces_plausible_pass_rates():
    config = EvalConfig(
        cases_path=Path("tests/fixtures/evaluation/phase01_mock_cases.jsonl"),
        backend_config_path=Path("configs/inference/mock.yaml"),
        predictions_path=PREDICTIONS_PATH,
        scorecard_path=None,
        case_results_path=None,
    )
    result = run_evaluation(config)
    scorecard = aggregate_scorecard(result.case_results)
    assert 0 <= scorecard.metrics.protocol_pass_rate <= 1
    assert 0 <= scorecard.metrics.effective_pass_rate <= 1
    assert scorecard.metrics.mean_task_correctness > 0
