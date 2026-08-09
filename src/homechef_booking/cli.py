"""HomeChef evaluation CLI entry point."""

import argparse
from pathlib import Path

import yaml

from homechef_booking.evaluation.runner import EvalConfig, run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="HomeChef evaluation harness")
    parser.add_argument("--config", type=str, required=True, help="Path to eval config YAML")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()
    config_path = Path(args.config)
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cases_path = Path(config_data["cases_path"])
    backend_path = Path(config_data["backend_config_path"])
    output_dir = Path(args.output_dir) if args.output_dir else Path(config_data.get("output_dir", "reports/generated"))
    predictions_path = Path(config_data["predictions_path"]) if "predictions_path" in config_data else None
    eval_config = EvalConfig(
        cases_path=cases_path,
        backend_config_path=backend_path,
        predictions_path=predictions_path,
        scorecard_path=output_dir / f"{config_data.get('run_id', 'phase01_mock')}_scorecard.json",
        case_results_path=output_dir / f"{config_data.get('run_id', 'phase01_mock')}_case_results.json",
    )
    result = run_evaluation(eval_config)
    print(f"Total cases: {result.scorecard.metrics.total_cases}")
    print(f"Protocol pass rate: {result.scorecard.metrics.protocol_pass_rate:.4f}")
    print(f"Effective pass rate: {result.scorecard.metrics.effective_pass_rate:.4f}")
    print(f"Mean task correctness: {result.scorecard.metrics.mean_task_correctness:.4f}")
    print(f"Critical error rate: {result.scorecard.metrics.critical_error_rate:.4f}")


if __name__ == "__main__":
    main()
