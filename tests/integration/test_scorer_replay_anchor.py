"""Task 9 & 10: Golden mock outcome replay anchor test.

Runs the full 19-case evaluation against mock predictions and checks every case
against the oracle. This is the integration gate — if this test fails, Phase 01
is not complete.
"""

import json
from pathlib import Path

from homechef_booking.evaluation.runner import EvalConfig, run_evaluation


CASES_PATH = Path("tests/fixtures/evaluation/phase01_mock_cases.jsonl")
PREDICTIONS_PATH = Path("tests/fixtures/evaluation/phase01_mock_predictions.json")
ORACLE_PATH = Path("tests/fixtures/evaluation/phase01_oracle.json")


def test_all_19_cases_match_oracle(tmp_path: Path):
    """Run full evaluation and check every case against the frozen oracle."""
    config = EvalConfig(
        cases_path=CASES_PATH,
        backend_config_path=Path("configs/inference/mock.yaml"),
        predictions_path=PREDICTIONS_PATH,
        scorecard_path=tmp_path / "scorecard.json",
        case_results_path=tmp_path / "case_results.json",
    )
    result = run_evaluation(config)

    oracle = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
    oracle_map = {case["id"]: case for case in oracle["cases"]}

    assert len(result.case_results) == 19, f"Expected 19 cases, got {len(result.case_results)}"

    mismatches = []
    for case_result in result.case_results:
        expected = oracle_map.get(case_result.id)
        if expected is None:
            mismatches.append(f"  {case_result.id}: MISSING from oracle")
            continue

        if case_result.effective_pass != expected["effective_pass"]:
            mismatches.append(
                f"  {case_result.id}: effective_pass: got {case_result.effective_pass}, expected {expected['effective_pass']}"
            )
        if sorted(case_result.critical_error_tags) != sorted(expected["critical_error_tags"]):
            mismatches.append(
                f"  {case_result.id}: critical_error_tags: got {sorted(case_result.critical_error_tags)}, expected {sorted(expected['critical_error_tags'])}"
            )
        if case_result.protocol_pass != expected["protocol_pass"]:
            mismatches.append(
                f"  {case_result.id}: protocol_pass: got {case_result.protocol_pass}, expected {expected['protocol_pass']}"
            )

        for metric_key, expected_val in expected["metric_outcomes"].items():
            actual_val = case_result.metric_outcomes.get(metric_key)
            if actual_val != expected_val:
                mismatches.append(
                    f"  {case_result.id}: metric_outcomes.{metric_key}: got {actual_val}, expected {expected_val}"
                )

    if mismatches:
        raise AssertionError(f"Oracle mismatch for {len(mismatches)} checks:\n" + "\n".join(mismatches))

    print(f"All 19 cases match oracle. effective_pass_rate={result.scorecard.metrics.effective_pass_rate:.4f}")
