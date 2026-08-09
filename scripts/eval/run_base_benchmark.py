"""Phase 02 M0 Base Benchmark runner.

Usage:
  uv run python scripts/eval/run_base_benchmark.py \
    --config configs/evaluation/phase02_base_0_6b_frozen.yaml \
    --config configs/evaluation/phase02_base_0_6b_diagnostic.yaml \
    --config configs/evaluation/phase02_base_1_7b_frozen.yaml \
    --config configs/evaluation/phase02_base_1_7b_diagnostic.yaml
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from homechef_booking.evaluation.benchmark_runner import BenchmarkConfig, run_benchmark
from homechef_booking.evaluation.data_card import generate_data_card
from homechef_booking.evaluation.failure_export import export_failure_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 02 M0 Base Benchmark")
    parser.add_argument("--config", type=str, action="append", required=True, help="Path to benchmark config YAML (repeatable)")
    parser.add_argument("--output-dir", type=str, default="reports/generated/phase02")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    all_results = []
    per_run_metadata = []

    for config_path_str in args.config:
        config_path = Path(config_path_str)
        bm_config = BenchmarkConfig.load_yaml(config_path)
        result = run_benchmark(bm_config)

        # Export failure report
        case_results_path = bm_config.output_dir / f"{bm_config.run_id}_case_results.json"
        failure_path = bm_config.output_dir / f"{bm_config.run_id}_failure_analysis.json"
        if case_results_path.exists():
            export_failure_report(case_results_path, failure_path, suite_id=bm_config.suite_id)

        # Generate data card
        data_card_path = bm_config.output_dir / f"{bm_config.run_id}_data_card.json"
        generate_data_card(result, bm_config, data_card_path)

        all_results.append(result)
        per_run_metadata.append({
            "run_id": result.run_id,
            "model_id": result.model_id,
            "suite_id": result.suite_id,
            "total_cases": result.total_cases,
            "protocol_pass_rate": result.protocol_pass_rate,
            "effective_pass_rate": result.effective_pass_rate,
            "mean_task_correctness": result.mean_task_correctness,
            "critical_error_rate": result.critical_error_rate,
            "scorecard_path": str(bm_config.output_dir / f"{bm_config.run_id}_scorecard.json"),
            "case_results_path": str(case_results_path),
            "failure_analysis_path": str(failure_path),
            "data_card_path": str(data_card_path),
            "benchmark_result_path": str(bm_config.output_dir / f"{bm_config.run_id}_benchmark.json"),
        })
        print(f"[{result.run_id}] effective_pass_rate={result.effective_pass_rate:.4f} mean_task_correctness={result.mean_task_correctness:.4f}")

    # Combined summary
    summary = {
        "benchmark_name": "HomeChef-Booking-M0",
        "phase": "02",
        "generated_at": datetime.now(UTC).isoformat(),
        "runs": per_run_metadata,
        "overall_summary": {
            "total_models": len(set(r.model_id for r in all_results)),
            "total_runs": len(all_results),
            "best_effective_pass_rate": max(r.effective_pass_rate for r in all_results),
            "best_mean_task_correctness": max(r.mean_task_correctness for r in all_results),
        },
    }

    summary_json_path = output_root / "base_benchmark_summary.json"
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nSummary written to {summary_json_path}")

    # Markdown summary
    md_lines = [
        "# Phase 02 M0 Base Benchmark Summary",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Per-Run Results",
        "",
        "| Run ID | Model | Suite | Cases | Protocol Pass | Effective Pass | Task Correctness | Critical Error |",
        "|--------|-------|-------|-------|---------------|----------------|------------------|----------------|",
    ]
    for meta in per_run_metadata:
        md_lines.append(
            f"| {meta['run_id']} | {meta['model_id']} | {meta['suite_id']} | {meta['total_cases']} | "
            f"{meta['protocol_pass_rate']:.4f} | {meta['effective_pass_rate']:.4f} | "
            f"{meta['mean_task_correctness']:.4f} | {meta['critical_error_rate']:.4f} |"
        )
    md_lines.extend([
        "",
        "## Overall",
        "",
        f"- Models tested: {summary['overall_summary']['total_models']}",
        f"- Total runs: {summary['overall_summary']['total_runs']}",
        f"- Best effective pass rate: {summary['overall_summary']['best_effective_pass_rate']:.4f}",
        f"- Best mean task correctness: {summary['overall_summary']['best_mean_task_correctness']:.4f}",
        "",
        "## Artifacts",
    ])
    for meta in per_run_metadata:
        md_lines.extend([
            f"### {meta['run_id']}",
            f"- Scorecard: `{meta['scorecard_path']}`",
            f"- Case Results: `{meta['case_results_path']}`",
            f"- Failure Analysis: `{meta['failure_analysis_path']}`",
            f"- Data Card: `{meta['data_card_path']}`",
            "",
        ])

    summary_md_path = output_root / "base_benchmark_summary.md"
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown summary written to {summary_md_path}")


if __name__ == "__main__":
    main()
