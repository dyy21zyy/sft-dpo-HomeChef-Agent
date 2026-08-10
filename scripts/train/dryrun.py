"""Phase 04 Task 3 Amendment 2 — Training dry-run script.

Refuses to run without --approval-file (Checkpoint B gate).
After approval, executes a real LLaMA-Factory engineering dry-run:
- SFT: 1 row, max_steps=2
- DPO: 1 pair, max_steps=2, pref_beta=0.1, pref_loss=sigmoid

Usage (after Checkpoint B approval):
  uv run python scripts/train/dryrun.py --stage sft --config configs/training/phase04_sft_qwen3_0_6b.yaml --sample-count 1 --max-steps 2 --approval-file project-log/phase04_dryrun_approval.json
  uv run python scripts/train/dryrun.py --stage dpo --config configs/training/phase04_dpo_dryrun_beta_0_1.yaml --sample-count 1 --max-steps 2 --approval-file project-log/phase04_dryrun_approval.json
"""

import argparse
import json
import subprocess
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path

from homechef_booking.training.config import load_training_run_spec, validate_training_run_spec
from homechef_booking.training.dataset_adapter import validate_dpo_for_training, validate_sft_for_training

# ── Constants ───────────────────────────────────────────────────────────────

DRYRUN_BASE = Path("experiments/phase04/dryrun")
MANIFEST_PATH = Path("project-log/phase04_dryrun_manifest.json")

SFT_TEMP_DATASET = Path("data") / "phase04_dryrun_sft_1row.jsonl"
SFT_TEMP_CONFIG = DRYRUN_BASE / "phase04_dryrun_sft_1row.yaml"
SFT_OUTPUT_DIR = DRYRUN_BASE / "sft_0_6b"

DPO_TEMP_DATASET = Path("data") / "phase04_dryrun_dpo_1pair.jsonl"
DPO_TEMP_CONFIG = DRYRUN_BASE / "phase04_dryrun_dpo_1pair.yaml"
DPO_OUTPUT_DIR = DRYRUN_BASE / "dpo_0_6b"


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


def _create_dryrun_dataset(source_path: Path, count: int, output_path: Path, label: str) -> Path:
    """Copy the first `count` rows from a JSONL dataset into a temp file."""
    lines = source_path.read_text(encoding="utf-8").splitlines()
    first_row = lines[0].strip()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(first_row + "\n", encoding="utf-8")
    print(f"  Temporary {label} dataset: {output_path} (1 row/pair)")
    return output_path


def _generate_dryrun_config(
    source_config: Path,
    output_path: Path,
    temp_dataset: Path,
    output_dir: Path,
    stage: str,
    extra_overrides: dict | None = None,
) -> Path:
    """Generate a temporary dry-run YAML config.

    Copies all SPEC fields from the source config and overrides:
    - train/eval dataset paths → temporary file
    - max_steps: 2
    - num_train_epochs: 1
    - save_total_limit: 1
    - output_dir
    - engineering_dryrun_only: true
    - per_device_train/eval_batch_size: 1
    """
    raw = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}

    # Map to LLaMA-Factory field names (dataset/eval_dataset, not train_dataset_path/eval_dataset_path)
    raw.pop("train_dataset_path", None)
    raw.pop("eval_dataset_path", None)
    if stage == "dpo":
        raw["dataset"] = "phase04_dryrun_dpo_1pair"
        raw["eval_dataset"] = "phase04_dryrun_dpo_1pair"
    else:
        raw["dataset"] = "phase04_dryrun_sft_1row"
        raw["eval_dataset"] = "phase04_dryrun_sft_1row"
    raw["max_steps"] = 2
    raw["num_train_epochs"] = 1
    raw["save_total_limit"] = 1
    raw["output_dir"] = str(output_dir)
    raw["per_device_train_batch_size"] = 1
    raw["per_device_eval_batch_size"] = 1
    # CPU-only environment: disable bf16, enable use_cpu
    raw["bf16"] = False
    raw["use_cpu"] = True

    if extra_overrides:
        raw.update(extra_overrides)

    # Remove fields not recognized by LLaMA-Factory
    for field in ("sample_count", "approval_required", "engineering_dryrun_only",
                  "mask_history", "train_on_prompt", "enable_thinking", "fp16"):
        raw.pop(field, None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(raw, default_flow_style=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"  Temporary config: {output_path}")
    return output_path


def _run_llamafactory_train(config_path: Path) -> tuple[int, str, str]:
    """Invoke llamafactory-cli train with the given config.

    Returns (exit_code, stdout, stderr).
    Uses sys.executable to find the correct venv, then locates llamafactory-cli
    in the same Scripts directory.
    """
    scripts_dir = Path(sys.executable).parent
    cli_path = scripts_dir / "llamafactory-cli.exe"
    if not cli_path.exists():
        cli_path = scripts_dir / "llamafactory-cli"
    if not cli_path.exists():
        print("  ERROR: llamafactory-cli not found in venv Scripts directory.")
        print(f"  Checked: {cli_path}")
        return 1, "", "llamafactory-cli not found in venv"
    cmd = [str(cli_path), "train", str(config_path)]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Write full stderr to a log file for debugging
    log_path = DRYRUN_BASE / "llamafactory_stderr.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if result.stderr:
        log_path.write_text(result.stderr, encoding="utf-8")
    return result.returncode, result.stdout, result.stderr


def _write_dryrun_manifest(
    sft_passed: bool,
    dpo_passed: bool,
    sft_sample_count: int,
    sft_max_steps: int,
    dpo_pair_count: int,
    dpo_max_steps: int,
    approval_file: str,
    sft_config: str,
    dpo_config: str,
    sft_train_dataset: str,
    sft_eval_dataset: str,
    dpo_train_dataset: str,
    dpo_eval_dataset: str,
    sft_stdout: str = "",
    sft_stderr: str = "",
    dpo_stdout: str = "",
    dpo_stderr: str = "",
) -> dict:
    """Write the dry-run manifest to project-log/phase04_dryrun_manifest.json."""
    manifest = {
        "sft_passed": sft_passed,
        "dpo_passed": dpo_passed,
        "sft_sample_count": sft_sample_count,
        "dpo_pair_count": dpo_pair_count,
        "sft_max_steps": sft_max_steps,
        "dpo_max_steps": dpo_max_steps,
        "checkpoint_b_approved": True,
        "approval_file": approval_file,
        "sft_config": sft_config,
        "dpo_config": dpo_config,
        "sft_temp_dataset": str(SFT_TEMP_DATASET),
        "sft_temp_config": str(SFT_TEMP_CONFIG),
        "sft_output_dir": str(SFT_OUTPUT_DIR),
        "dpo_temp_dataset": str(DPO_TEMP_DATASET),
        "dpo_temp_config": str(DPO_TEMP_CONFIG),
        "dpo_output_dir": str(DPO_OUTPUT_DIR),
        "sft_train_dataset": sft_train_dataset,
        "sft_eval_dataset": sft_eval_dataset,
        "dpo_train_dataset": dpo_train_dataset,
        "dpo_eval_dataset": dpo_eval_dataset,
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


def _run_sft_dryrun(config_path: Path, spec, approval_file: str, sample_count: int, max_steps: int) -> None:
    """Execute SFT dry-run: create temp dataset, config, invoke LLaMA-Factory."""
    print(f"\nPreparing SFT dry-run:")
    print(f"  sample_count: {sample_count}")
    print(f"  max_steps: {max_steps}")

    sft_train = spec.train_dataset_path
    if sft_train is None:
        print("ERROR: No train dataset path in config.")
        sys.exit(1)

    _create_dryrun_dataset(sft_train, sample_count, SFT_TEMP_DATASET, "SFT")
    _generate_dryrun_config(config_path, SFT_TEMP_CONFIG, SFT_TEMP_DATASET, SFT_OUTPUT_DIR, "sft")
    exit_code, stdout, stderr = _run_llamafactory_train(SFT_TEMP_CONFIG)

    sft_passed = exit_code == 0
    if sft_passed:
        print(f"\nSFT dry-run PASSED.")
    else:
        print(f"\nSFT dry-run FAILED (exit code: {exit_code}).")
        if stderr:
            print(f"  stderr: {stderr[:2000]}")

    _write_dryrun_manifest(
        sft_passed=sft_passed,
        dpo_passed=False,
        sft_sample_count=sample_count,
        sft_max_steps=max_steps,
        dpo_pair_count=0,
        dpo_max_steps=0,
        approval_file=approval_file,
        sft_config=str(config_path),
        dpo_config="configs/training/phase04_dpo_dryrun_beta_0_1.yaml",
        sft_train_dataset=str(sft_train),
        sft_eval_dataset=str(spec.eval_dataset_path) if spec.eval_dataset_path else "",
        dpo_train_dataset="data/processed/phase03_dpo_targeted_v0.1_train.jsonl",
        dpo_eval_dataset="data/processed/phase03_dpo_targeted_v0.1_val.jsonl",
        sft_stdout=stdout,
        sft_stderr=stderr,
    )

    if not sft_passed:
        sys.exit(exit_code)


def _run_dpo_dryrun(config_path: Path, spec, approval_file: str, sample_count: int, max_steps: int) -> None:
    """Execute DPO dry-run: create temp dataset, config, invoke LLaMA-Factory."""
    print(f"\nPreparing DPO dry-run:")
    print(f"  pair_count: {sample_count}")
    print(f"  max_steps: {max_steps}")

    dpo_train = spec.train_dataset_path
    if dpo_train is None:
        print("ERROR: No train dataset path in config.")
        sys.exit(1)

    _create_dryrun_dataset(dpo_train, sample_count, DPO_TEMP_DATASET, "DPO")
    _generate_dryrun_config(
        config_path,
        DPO_TEMP_CONFIG,
        DPO_TEMP_DATASET,
        DPO_OUTPUT_DIR,
        "dpo",
        extra_overrides={"pref_beta": 0.1, "pref_loss": "sigmoid"},
    )
    exit_code, stdout, stderr = _run_llamafactory_train(DPO_TEMP_CONFIG)

    dpo_passed = exit_code == 0
    if dpo_passed:
        print(f"\nDPO dry-run PASSED.")
    else:
        print(f"\nDPO dry-run FAILED (exit code: {exit_code}).")
        if stderr:
            print(f"  stderr: {stderr[:2000]}")

    _write_dryrun_manifest(
        sft_passed=False,
        dpo_passed=dpo_passed,
        sft_sample_count=0,
        sft_max_steps=0,
        dpo_pair_count=sample_count,
        dpo_max_steps=max_steps,
        approval_file=approval_file,
        sft_config="configs/training/phase04_sft_qwen3_0_6b.yaml",
        dpo_config=str(config_path),
        sft_train_dataset="data/processed/phase03_sft_v0.1_train.jsonl",
        sft_eval_dataset="data/processed/phase03_sft_v0.1_val.jsonl",
        dpo_train_dataset=str(dpo_train),
        dpo_eval_dataset=str(spec.eval_dataset_path) if spec.eval_dataset_path else "",
        dpo_stdout=stdout,
        dpo_stderr=stderr,
    )

    if not dpo_passed:
        sys.exit(exit_code)


def main() -> None:
    parser = argparse.ArgumentParser(description="Training dry-run (approval-gated, executes LLaMA-Factory)")
    parser.add_argument("--stage", required=True, choices=["sft", "dpo"], type=str)
    parser.add_argument("--config", required=True, type=str, help="Path to training YAML config")
    parser.add_argument("--sample-count", required=True, type=int, help="Number of samples/pairs to use")
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
    if args.stage == "dpo":
        if spec.train_dataset_path:
            train_report = validate_dpo_for_training(spec.train_dataset_path, expected_count=216)
            if not train_report.passed:
                print("DPO train dataset validation FAILED:")
                for err in train_report.errors:
                    print(f"  - {err}")
                sys.exit(1)
            print(f"  Train dataset OK: {train_report.total_rows} pairs")
        if spec.eval_dataset_path:
            eval_report = validate_dpo_for_training(spec.eval_dataset_path, expected_count=24)
            if not eval_report.passed:
                print("DPO eval dataset validation FAILED:")
                for err in eval_report.errors:
                    print(f"  - {err}")
                sys.exit(1)
            print(f"  Eval dataset OK: {eval_report.total_rows} pairs")
    else:
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

    # ── Execute dry-run ──
    if args.stage == "dpo":
        _run_dpo_dryrun(config_path, spec, args.approval_file, args.sample_count, args.max_steps)
    else:
        _run_sft_dryrun(config_path, spec, args.approval_file, args.sample_count, args.max_steps)


if __name__ == "__main__":
    main()
