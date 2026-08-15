"""Phase 02 M0 Base Benchmark runner.

Generates combined summary from existing benchmark artifacts.
Does NOT re-run inference — reads from saved JSON files only.

Usage:
  uv run python scripts/eval/run_base_benchmark.py \
    --config configs/evaluation/phase02_base_0_6b_frozen.yaml \
    --config configs/evaluation/phase02_base_1_7b_frozen.yaml \
    --config configs/evaluation/phase02_base_4b_frozen.yaml
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value / 1000:.2f}s"
    return f"{value:.0f}ms"


def _fmt_tps(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 02 M0 Base Benchmark Summary")
    parser.add_argument("--config", type=str, action="append", required=True,
                        help="Path to benchmark config YAML (repeatable)")
    parser.add_argument("--output-dir", type=str, default="reports/generated/phase02")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    per_run_metadata = []

    for config_path_str in args.config:
        config_path = Path(config_path_str)
        import yaml
        config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        run_id = config_data["run_id"]
        output_dir = Path(config_data.get("output_dir", "reports/generated"))
        benchmark_path = output_dir / f"{run_id}_benchmark.json"

        if not benchmark_path.exists():
            print(f"[SKIP] {run_id}: benchmark result not found at {benchmark_path}")
            continue

        result = json.loads(benchmark_path.read_text(encoding="utf-8"))
        lm = result.get("latency_metrics") or {}

        meta = {
            "run_id": run_id,
            "model_id": result.get("model_id", "unknown"),
            "suite_id": result.get("suite_id", ""),
            "total_cases": result.get("total_cases", 0),
            "protocol_pass_rate": result.get("protocol_pass_rate"),
            "effective_pass_rate": result.get("effective_pass_rate"),
            "mean_task_correctness": result.get("mean_task_correctness"),
            "critical_error_rate": result.get("critical_error_rate"),
            "mean_latency_ms": lm.get("mean_latency_ms"),
            "p95_latency_ms": lm.get("p95_latency_ms"),
            "mean_ttft_ms": lm.get("mean_ttft_ms"),
            "mean_tokens_per_second": lm.get("mean_tokens_per_second"),
            "performance_sample_count": lm.get("performance_sample_count", 0),
            "failed_inference_cases": lm.get("failed_inference_cases", 0),
            "timeout_cases": lm.get("timeout_cases", 0),
            "total_wall_seconds": lm.get("total_wall_seconds", 0),
            "scorecard_path": str(output_dir / f"{run_id}_scorecard.json"),
            "case_results_path": str(output_dir / f"{run_id}_case_results.json"),
            "benchmark_result_path": str(benchmark_path),
            "hardware_info": result.get("hardware_info"),
            "quantization": config_data.get("quantization", ""),
        }
        per_run_metadata.append(meta)
        print(f"[{run_id}] effective_pass={_fmt_rate(meta['effective_pass_rate'])} "
              f"latency={_fmt_ms(meta['mean_latency_ms'])} "
              f"ttft={_fmt_ms(meta['mean_ttft_ms'])} "
              f"tps={_fmt_tps(meta['mean_tokens_per_second'])}")

    # Combined summary JSON
    summary = {
        "benchmark_name": "HomeChef-Booking-M0",
        "phase": "02",
        "runtime": "Windows CPU + GGUF + llama.cpp + llama-server",
        "generated_at": datetime.now(UTC).isoformat(),
        "runs": per_run_metadata,
        "overall_summary": {
            "total_models": len(set(r["model_id"] for r in per_run_metadata)),
            "total_runs": len(per_run_metadata),
        },
    }

    summary_json_path = output_root / "base_benchmark_summary.json"
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nSummary written to {summary_json_path}")

    # Markdown summary
    md_lines = [
        "# Phase 02 M0 Base Benchmark",
        "",
        "**Runtime:** Windows CPU + GGUF + llama.cpp + llama-server (OpenAI-compatible HTTP API)",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Core Results",
        "",
        "| Model | Protocol | Task Correctness | Effective Pass | Avg Latency | P95 | Avg TTFT | Throughput |",
        "|-------|---------:|-----------------:|---------------:|------------:|----:|---------:|-----------:|",
    ]

    for meta in per_run_metadata:
        eff_str = f"{meta['effective_pass_rate']:.4f}" if meta['effective_pass_rate'] is not None else "—"
        if meta.get("total_cases", 0) > 0 and meta['effective_pass_rate'] is not None:
            eff_count = int(round(meta['effective_pass_rate'] * meta['total_cases']))
            eff_str = f"{eff_count}/{meta['total_cases']} ({eff_str})"

        md_lines.append(
            f"| {meta['model_id']} "
            f"| {_fmt_rate(meta['protocol_pass_rate'])} "
            f"| {_fmt_rate(meta['mean_task_correctness'])} "
            f"| {eff_str} "
            f"| {_fmt_ms(meta['mean_latency_ms'])} "
            f"| {_fmt_ms(meta['p95_latency_ms'])} "
            f"| {_fmt_ms(meta['mean_ttft_ms'])} "
            f"| {_fmt_tps(meta['mean_tokens_per_second'])} |"
        )

    md_lines.extend([
        "",
        "## Performance Details",
        "",
        "| Model | Perf Samples | Failed | Timeout | Wall Time | Quantization |",
        "|-------|-------------:|-------:|--------:|----------:|-------------:|",
    ])
    for meta in per_run_metadata:
        wall_str = f"{meta['total_wall_seconds']:.0f}s" if meta.get("total_wall_seconds") else "—"
        md_lines.append(
            f"| {meta['model_id']} "
            f"| {meta['performance_sample_count']} "
            f"| {meta['failed_inference_cases']} "
            f"| {meta.get('timeout_cases', 0)} "
            f"| {wall_str} "
            f"| {meta.get('quantization', '—')} |"
        )

    md_lines.extend([
        "",
        "## Hardware",
        "",
        "| Model | OS | CPU | Runtime | GPU Layers |",
        "|-------|----|-----|---------|-----------:|",
    ])
    for meta in per_run_metadata:
        hw = meta.get("hardware_info") or {}
        md_lines.append(
            f"| {meta['model_id']} "
            f"| {hw.get('platform', '—')} "
            f"| {hw.get('device', '—')} "
            f"| {hw.get('runtime', '—')} "
            f"| {hw.get('gpu_layers', '—')} |"
        )

    md_lines.extend([
        "",
        "## Notes",
        "",
        "- Phase 02 uses Qwen3 Base GGUF + llama.cpp CPU for pre-training baseline benchmark.",
        "- Runtime is aligned with local CPU deployment path (no CUDA, no HF Transformers inference).",
        "- Diagnostic Dev is preserved but excluded from the current formal matrix.",
        "- Each model × 120 Frozen Test cases = 360 total inferences.",
        "- Latency = model HTTP inference time only (not E2E evaluation wall time).",
        "- TTFT measured via streaming (stream=true).",
        "- Throughput = decode tokens/second = completion_tokens / (latency - ttft).",
    ])

    summary_md_path = output_root / "base_benchmark_summary.md"
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown summary written to {summary_md_path}")


if __name__ == "__main__":
    main()
