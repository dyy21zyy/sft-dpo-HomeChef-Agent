"""Phase 04 Task 3 — Training dry-run script.

Refuses to run without --approval-file (Checkpoint B gate).
Validates config and datasets, then reports readiness without executing training.

Usage (after Checkpoint B approval):
  uv run python scripts/train/dryrun.py --stage sft --config configs/training/phase04_sft_qwen3_0_6b.yaml --sample-count 1 --max-steps 2 --approval-file project-log/phase04_dryrun_approval.json
"""

import argparse
import json
import sys
from pathlib import Path

from homechef_booking.training.config import load_training_run_spec, validate_training_run_spec
from homechef_booking.training.dataset_adapter import validate_dpo_for_training, validate_sft_for_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Training dry-run (approval-gated)")
    parser.add_argument("--stage", required=True, choices=["sft", "dpo"], type=str)
    parser.add_argument("--config", required=True, type=str, help="Path to training YAML config")
    parser.add_argument("--sample-count", required=True, type=int, help="Number of samples to use")
    parser.add_argument("--max-steps", required=True, type=int, help="Max training steps")
    parser.add_argument("--approval-file", type=str, default=None,
                        help="Path to Checkpoint B approval JSON (required to proceed)")
    args = parser.parse_args()

    # Checkpoint B: refuse without approval file
    if not args.approval_file:
        print("ERROR: --approval-file is required. Checkpoint B is not approved.")
        print("  Dry-run cannot execute without approval.")
        sys.exit(1)

    approval_path = Path(args.approval_file)
    if not approval_path.exists():
        print(f"ERROR: Approval file not found: {args.approval_file}")
        print("  Checkpoint B is not approved. Create project-log/phase04_dryrun_approval.json")
        sys.exit(1)

    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        if not approval.get("approved"):
            print("ERROR: Approval file exists but 'approved' is not true.")
            print("  Checkpoint B is not approved.")
            sys.exit(1)
    except json.JSONDecodeError:
        print("ERROR: Approval file is not valid JSON.")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    spec = load_training_run_spec(config_path)
    config_errors = validate_training_run_spec(spec)
    if config_errors:
        print("Config validation FAILED:")
        for err in config_errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"Config loaded: stage={spec.stage}, model={spec.model_name_or_path}")
    print(f"  Train dataset: {spec.train_dataset_path}")
    print(f"  Eval dataset: {spec.eval_dataset_path}")

    # Validate datasets
    if spec.train_dataset_path:
        if spec.stage == "dpo":
            train_report = validate_dpo_for_training(spec.train_dataset_path, expected_count=216)
        else:
            train_report = validate_sft_for_training(spec.train_dataset_path, expected_count=540)
        if not train_report.passed:
            print(f"Train dataset validation FAILED:")
            for err in train_report.errors:
                print(f"  - {err}")
            sys.exit(1)
        print(f"  Train dataset OK: {train_report.total_rows} rows")

    if spec.eval_dataset_path:
        if spec.stage == "dpo":
            eval_report = validate_dpo_for_training(spec.eval_dataset_path, expected_count=24)
        else:
            eval_report = validate_sft_for_training(spec.eval_dataset_path, expected_count=60)
        if not eval_report.passed:
            print(f"Eval dataset validation FAILED:")
            for err in eval_report.errors:
                print(f"  - {err}")
            sys.exit(1)
        print(f"  Eval dataset OK: {eval_report.total_rows} rows")

    print(f"\nDry-run parameters:")
    print(f"  sample_count: {args.sample_count}")
    print(f"  max_steps: {args.max_steps}")
    print(f"\nDry-run validation PASSED.")
    print(f"  Stage: {spec.stage}")
    print(f"  Model: {spec.model_name_or_path}")
    print(f"  Ready for training when Checkpoint B is approved.")


if __name__ == "__main__":
    main()
