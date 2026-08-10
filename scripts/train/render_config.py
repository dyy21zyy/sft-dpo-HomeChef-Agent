"""Phase 04 Task 2 — Render a resolved training config copy.

Validates a training YAML config, validates the referenced dataset files,
and writes a resolved copy under experiments/phase04/configs/.

Usage:
  uv run python scripts/train/render_config.py --config configs/training/phase04_sft_qwen3_0_6b.yaml --output experiments/phase04/configs/phase04_sft_qwen3_0_6b.resolved.yaml
"""

import argparse
from pathlib import Path

import yaml

from homechef_booking.training.config import load_training_run_spec, validate_training_run_spec
from homechef_booking.training.dataset_adapter import validate_dpo_for_training, validate_sft_for_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a resolved training config")
    parser.add_argument("--config", required=True, type=str, help="Path to training YAML config")
    parser.add_argument("--output", required=True, type=str, help="Output path for resolved YAML")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_path = Path(args.output)

    # Step 1: Load and validate config
    spec = load_training_run_spec(config_path)
    config_errors = validate_training_run_spec(spec)
    if config_errors:
        print(f"Config validation FAILED for {config_path}:")
        for err in config_errors:
            print(f"  - {err}")
        raise SystemExit(1)

    # Step 2: Validate dataset files
    if spec.train_dataset_path:
        if spec.stage == "dpo":
            train_report = validate_dpo_for_training(spec.train_dataset_path, expected_count=216)
        else:
            train_report = validate_sft_for_training(spec.train_dataset_path, expected_count=540)
        if not train_report.passed:
            print(f"Train dataset validation FAILED for {spec.train_dataset_path}:")
            for err in train_report.errors:
                print(f"  - {err}")
            raise SystemExit(1)
        print(f"Train dataset OK: {train_report.total_rows} rows")

    if spec.eval_dataset_path:
        if spec.stage == "dpo":
            eval_report = validate_dpo_for_training(spec.eval_dataset_path, expected_count=24)
        else:
            eval_report = validate_sft_for_training(spec.eval_dataset_path, expected_count=60)
        if not eval_report.passed:
            print(f"Eval dataset validation FAILED for {spec.eval_dataset_path}:")
            for err in eval_report.errors:
                print(f"  - {err}")
            raise SystemExit(1)
        print(f"Eval dataset OK: {eval_report.total_rows} rows")

    # Step 3: Write resolved config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = spec.model_dump(mode="json", exclude_none=True)
    # Convert Path fields to strings for YAML output
    for key in ("train_dataset_path", "eval_dataset_path", "adapter_name_or_path"):
        if key in resolved and resolved[key] is not None:
            resolved[key] = str(resolved[key])
    output_path.write_text(yaml.dump(resolved, default_flow_style=False, allow_unicode=True), encoding="utf-8")
    print(f"Resolved config written to {output_path}")


if __name__ == "__main__":
    main()
