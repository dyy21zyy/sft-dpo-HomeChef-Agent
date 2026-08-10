"""Phase 04 Task 3 Amendment — Training dry-run script.

Refuses to run without --approval-file (Checkpoint B gate).
After approval, executes a real LLaMA-Factory engineering dry-run:
- SFT: 1 row, max_steps=2
- DPO: BLOCKED until dpo_beta_field_name is confirmed

Usage (after Checkpoint B approval):
  uv run python scripts/train/dryrun.py --stage sft --config configs/training/phase04_sft_qwen3_0_6b.yaml --sample-count 1 --max-steps 2 --approval-file project-log/phase04_dryrun_approval.json
"""

import argparse
import json
import subprocess
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path

from homechef_booking.training.config import load_training_run_spec, validate_training_run_spec
from homechef_booking.training.dataset_adapter import validate_sft_for_training

# ── Constants ───────────────────────────────────────────────────────────────

DRYRUN_BASE = Path("experiments/phase04/dryrun")
MANIFEST_PATH = Path("project-log/phase04_dryrun_manifest.json")

SFT_TEMP_DATASET = DRYRUN_BASE / "phase04_dryrun_sft_1row.jsonl"
SFT_TEMP_CONFIG = DRYRUN_BASE / "phase04_dryrun_sft_1row.yaml"
SFT_OUTPUT_DIR = DRYRUN_BASE / "sft_0_6b"


def _validate_approval(approval_path: Path) -> dict:
    """Validate the Checkpoint B approval file. Returns the parsed JSON."""
    if not approval_path.exists():
        print(f"ERROR: Approval file not found: {approval_path}")
        print("  Checkpoint B is not approved.")
        sys.exit(1)
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("ERROR: Approval file is not valid JSON.")
        sys.exit(1)
    if not approval.get("approved"):
        print("ERROR: Approval file exists but 'approved' is not true.")
        print("  Checkpoint B is not approved.")
        sys.exit(1)
    return approval


def _create_sft_dryrun_dataset(source_path: Path, count: int) -> Path:
    """Copy the first `count` rows from the SFT train dataset into a temp file."""
    lines = source_path.read_text(encoding="utf-8").splitlines()
    first_row = lines[0].strip()
    SFT_TEMP_DATASET.parent.mkdir(parents=True, exist_ok=True)
    SFT_TEMP_DATASET.write_text(first_row + "\n", encoding="utf-8")
    print(f"  Temporary SFT dataset: {SFT_TEMP_DATASET} (1 row)")
    return SFT_TEMP_DATASET


def _generate_sft_dryrun_config(source_config: Path, spec) -> Path:
    """Generate a temporary SFT dry-run YAML config.

    Copies all SPEC fields from the source config and overrides:
    - train/eval dataset paths → temporary 1-row file
    - max_steps: 2
    - num_train_epochs: 1
    - save_total_limit: 1
    - output_dir: experiments/phase04/dryrun/sft_0_6b/
    - engineering_dryrun_only: true
    """
    raw = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}

    # Override for dry-run
    raw["train_dataset_path"] = str(SFT_TEMP_DATASET)
    raw["eval_dataset_path"] = str(SFT_TEMP_DATASET)
    raw["max_steps"] = 2
    raw["num_train_epochs"] = 1
    raw["save_total_limit"] = 1
    raw["output_dir"] = str(SFT_OUTPUT_DIR)
    raw["engineering_dryrun_only"] = True
    raw["per_device_train_batch_size"] = 1
    raw["per_device_eval_batch_size"] = 1

    # Remove formal-only fields that conflict with dry-run
    raw.pop("sample_count", None)
    raw.pop("approval_required", None)

    SFT_TEMP_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    SFT_TEMP_CONFIG.write_text(
        yaml.dump(raw, default_flow_style=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"  Temporary SFT config: {SFT_TEMP_CONFIG}")
    return SFT_TEMP_CONFIG


def _run_llamafactory_train(config_path: Path) -> tuple[int, str, str]:
    """Invoke llamafactory-cli train with the given config.

    Returns (exit_code, stdout, stderr).
    """
    cmd = ["llamafactory-cli", "train", str(config_path)]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def _write_dryrun_manifest(
    sft_passed: bool,
    dpo_passed: bool,
    dpo_blocked_reason: str,
    sft_sample_count: int,
    sft_max_steps: int,
    approval_file: str,
    sft_config: str,
    sft_train_dataset: str,
    sft_eval_dataset: str,
    sft_stdout: str = "",
    sft_stderr: str = "",
) -> dict:
    """Write the dry-run manifest to project-log/phase04_dryrun_manifest.json."""
    manifest = {
        "sft_passed": sft_passed,
        "dpo_passed": dpo_passed,
        "dpo_blocked_reason": dpo_blocked_reason,
        "sft_sample_count": sft_sample_count,
        "dpo_pair_count": 1,
        "sft_max_steps": sft_max_steps,
        "dpo_max_steps": 2,
        "checkpoint_b_approved": True,
        "approval_file": approval_file,
        "sft_config": sft_config,
        "dpo_config": "configs/training/phase04_dpo_dryrun_beta_0_1.yaml",
        "sft_temp_dataset": str(SFT_TEMP_DATASET),
        "sft_temp_config": str(SFT_TEMP_CONFIG),
        "sft_output_dir": str(SFT_OUTPUT_DIR),
        "sft_train_dataset": sft_train_dataset,
        "sft_eval_dataset": sft_eval_dataset,
        "dpo_train_dataset": "data/processed/phase03_dpo_targeted_v0.1_train.jsonl",
        "dpo_eval_dataset": "data/processed/phase03_dpo_targeted_v0.1_val.jsonl",
        "frozen_test_used_for_training": False,
        "diagnostic_dev_used_for_training": False,
        "phase02_output_used_for_training": False,
        "no_training_effectiveness_claim": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDry-run manifest written to {MANIFEST_PATH}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Training dry-run (approval-gated, executes LLaMA-Factory)")
    parser.add_argument("--stage", required=True, choices=["sft", "dpo"], type=str)
    parser.add_argument("--config", required=True, type=str, help="Path to training YAML config")
    parser.add_argument("--sample-count", required=True, type=int, help="Number of samples to use")
    parser.add_argument("--max-steps", required=True, type=int, help="Max training steps")
    parser.add_argument("--approval-file", type=str, default=None,
                        help="Path to Checkpoint B approval JSON (required to proceed)")
    args = parser.parse_args()

    # ── Gate 1: Approval file ──
    if not args.approval_file:
        print("ERROR: --approval-file is required. Checkpoint B is not approved.")
        print("  Dry-run cannot execute without approval.")
        sys.exit(1)

    approval = _validate_approval(Path(args.approval_file))

    # ── Gate 2: Source config ──
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    spec = load_training_run_spec(config_path)

    # DPO is blocked before any validation
    if args.stage == "dpo":
        print(f"Config loaded: stage={spec.stage}, model={spec.model_name_or_path}")
        print("\nDPO dry-run is BLOCKED.")
        print("  Reason: dpo_beta_field_name is not confirmed.")
        print("  The exact LLaMA-Factory config key for DPO preference beta must be verified.")
        print("  Skipping DPO dry-run.")

        _write_dryrun_manifest(
            sft_passed=False,
            dpo_passed=False,
            dpo_blocked_reason="dpo_beta_field_name_unconfirmed",
            sft_sample_count=0,
            sft_max_steps=0,
            approval_file=args.approval_file,
            sft_config=args.config,
            sft_train_dataset=str(spec.train_dataset_path) if spec.train_dataset_path else "",
            sft_eval_dataset=str(spec.eval_dataset_path) if spec.eval_dataset_path else "",
        )
        sys.exit(0)

    config_errors = validate_training_run_spec(spec)
    if config_errors:
        print("Config validation FAILED:")
        for err in config_errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"Config loaded: stage={spec.stage}, model={spec.model_name_or_path}")
    print(f"  Train dataset: {spec.train_dataset_path}")
    print(f"  Eval dataset: {spec.eval_dataset_path}")

    # ── Gate 3: Validate source datasets ──
    if spec.train_dataset_path:
        train_report = validate_sft_for_training(spec.train_dataset_path, expected_count=540)
        if not train_report.passed:
            print("Train dataset validation FAILED:")
            for err in train_report.errors:
                print(f"  - {err}")
            sys.exit(1)
        print(f"  Train dataset OK: {train_report.total_rows} rows")

    if spec.eval_dataset_path:
        eval_report = validate_sft_for_training(spec.eval_dataset_path, expected_count=60)
        if not eval_report.passed:
            print("Eval dataset validation FAILED:")
            for err in eval_report.errors:
                print(f"  - {err}")
            sys.exit(1)
        print(f"  Eval dataset OK: {eval_report.total_rows} rows")

    # ── SFT dry-run execution ──
    print(f"\nPreparing SFT dry-run:")
    print(f"  sample_count: {args.sample_count}")
    print(f"  max_steps: {args.max_steps}")

    # Step 1: Create 1-row temporary dataset
    sft_train = spec.train_dataset_path
    if sft_train is None:
        print("ERROR: No train dataset path in config.")
        sys.exit(1)
    _create_sft_dryrun_dataset(sft_train, args.sample_count)

    # Step 2: Generate temporary dry-run config
    _generate_sft_dryrun_config(config_path, spec)

    # Step 3: Invoke LLaMA-Factory
    exit_code, stdout, stderr = _run_llamafactory_train(SFT_TEMP_CONFIG)

    sft_passed = exit_code == 0
    if sft_passed:
        print(f"\nSFT dry-run PASSED.")
    else:
        print(f"\nSFT dry-run FAILED (exit code: {exit_code}).")
        if stderr:
            print(f"  stderr: {stderr[:2000]}")

    # Step 4: Write manifest
    _write_dryrun_manifest(
        sft_passed=sft_passed,
        dpo_passed=False,
        dpo_blocked_reason="dpo_beta_field_name_unconfirmed",
        sft_sample_count=args.sample_count,
        sft_max_steps=args.max_steps,
        approval_file=args.approval_file,
        sft_config=args.config,
        sft_train_dataset=str(sft_train),
        sft_eval_dataset=str(spec.eval_dataset_path) if spec.eval_dataset_path else "",
        sft_stdout=stdout,
        sft_stderr=stderr,
    )

    if not sft_passed:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
