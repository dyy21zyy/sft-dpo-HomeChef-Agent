"""Phase 04 Task 3 Amendment 3 — Training dry-run script.

Refuses to run without --approval-file (Checkpoint B gate).
After approval, executes a real LLaMA-Factory engineering dry-run:
- SFT: 1 row, max_steps=2
- DPO: 1 pair, max_steps=2, pref_beta=0.1, pref_loss=sigmoid

Temp datasets go under experiments/phase04/dryrun/.
DPO pairs are converted to LLaMA-Factory 0.9.5 standard format.

Usage (after Checkpoint B approval):
  python scripts/train/dryrun.py --stage sft --config configs/training/phase04_sft_qwen3_0_6b.yaml --sample-count 1 --max-steps 2 --approval-file project-log/phase04_dryrun_approval.json
  python scripts/train/dryrun.py --stage dpo --config configs/training/phase04_dpo_dryrun_beta_0_1.yaml --sample-count 1 --max-steps 2 --approval-file project-log/phase04_dryrun_approval.json
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

# Temp datasets go under data/ because LLaMA-Factory resolves file_name relative to
# dataset_dir=data/ and cannot handle non-ASCII paths. They are clearly named
# phase04_dryrun_* to distinguish from permanent datasets.
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


def _convert_dpo_pair_to_llamafactory_format(pair: dict) -> dict:
    """Convert a DPO pair to LLaMA-Factory 0.9.5 compatible ShareGPT ranking format.

    LLaMA-Factory's SharegptDatasetConverter enforces strict alternating tags
    (user/observation on odd positions, assistant/function_call on even positions).
    Multi-turn conversations with tool calls violate this pattern.

    For the engineering dry-run, we serialize the full prompt context into a single
    user message, producing a clean single-turn shape:

    {
      "conversations": [{"from": "user", "value": "<serialized prompt>"}],
      "chosen": {"from": "assistant", "value": "<chosen JSON>"},
      "rejected": {"from": "assistant", "value": "<rejected JSON>"}
    }
    """
    prompt = pair.get("prompt", [])

    # Serialize the full prompt context into one user message
    serialized_prompt = json.dumps(prompt, ensure_ascii=False)
    conversations = [{"from": "user", "value": serialized_prompt}]

    chosen_raw = pair.get("chosen", "")
    rejected_raw = pair.get("rejected", "")

    return {
        "conversations": conversations,
        "chosen": {"from": "assistant", "value": chosen_raw},
        "rejected": {"from": "assistant", "value": rejected_raw},
    }


def _create_dpo_dryrun_dataset(source_path: Path, count: int) -> Path:
    """Copy the first DPO pair, convert to LLaMA-Factory format, write to temp file.

    Also updates data/dataset_info.json with the absolute path to the temp file
    so LLaMA-Factory can find it.
    """
    lines = source_path.read_text(encoding="utf-8").splitlines()
    first_row = json.loads(lines[0].strip())
    converted = _convert_dpo_pair_to_llamafactory_format(first_row)
    DPO_TEMP_DATASET.parent.mkdir(parents=True, exist_ok=True)
    DPO_TEMP_DATASET.write_text(json.dumps(converted, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  Temporary DPO dataset: {DPO_TEMP_DATASET} (1 pair, LLaMA-Factory format)")

    # Register in dataset_info.json with relative path from data/
    _register_temp_dataset("phase04_dryrun_dpo_1pair", DPO_TEMP_DATASET, is_dpo=True)

    return DPO_TEMP_DATASET


def _create_sft_dryrun_dataset(source_path: Path, count: int) -> Path:
    """Copy the first `count` rows from the SFT train dataset into a temp file.

    Also updates data/dataset_info.json with the absolute path.
    """
    lines = source_path.read_text(encoding="utf-8").splitlines()
    first_row = lines[0].strip()
    SFT_TEMP_DATASET.parent.mkdir(parents=True, exist_ok=True)
    SFT_TEMP_DATASET.write_text(first_row + "\n", encoding="utf-8")
    print(f"  Temporary SFT dataset: {SFT_TEMP_DATASET} (1 row)")

    _register_temp_dataset("phase04_dryrun_sft_1row", SFT_TEMP_DATASET, is_dpo=False)

    return SFT_TEMP_DATASET


def _register_temp_dataset(name: str, temp_path: Path, is_dpo: bool) -> None:
    """Register a temp dry-run dataset in data/dataset_info.json.

    Uses a relative path from data/ since LLaMA-Factory resolves file_name
    relative to dataset_dir (data/). The temp file must be under data/ to
    avoid non-ASCII path encoding issues with LLaMA-Factory.
    """
    info_path = Path("data/dataset_info.json")
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}

    # Use relative path from data/ — temp file is directly under data/
    relative_path = temp_path.name

    entry: dict = {
        "file_name": relative_path,
        "formatting": "sharegpt",
    }

    if is_dpo:
        entry.update({
            "ranking": True,
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        })
    else:
        entry.update({
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        })

    info[name] = entry
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def _generate_dryrun_config(
    source_config: Path,
    output_path: Path,
    output_dir: Path,
    stage: str,
    extra_overrides: dict | None = None,
) -> Path:
    """Generate a temporary dry-run YAML config."""
    raw = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}

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
    raw["bf16"] = False
    raw["use_cpu"] = True

    if extra_overrides:
        raw.update(extra_overrides)

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
    """Invoke llamafactory-cli train with the given config."""
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
    log_path = DRYRUN_BASE / "llamafactory_stderr.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if result.stderr:
        log_path.write_text(result.stderr, encoding="utf-8")
    return result.returncode, result.stdout, result.stderr


def _load_existing_manifest() -> dict:
    """Load existing manifest if present, otherwise return empty base dict."""
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "checkpoint_b_approved": True,
        "frozen_test_used_for_training": False,
        "diagnostic_dev_used_for_training": False,
        "phase02_output_used_for_training": False,
        "no_training_effectiveness_claim": True,
    }


def _update_sft_manifest(
    sft_passed: bool,
    sft_sample_count: int,
    sft_max_steps: int,
    approval_file: str,
    sft_config: str,
    sft_train_dataset: str,
    sft_eval_dataset: str,
) -> dict:
    """Update SFT fields in the dry-run manifest, preserving DPO fields."""
    manifest = _load_existing_manifest()
    manifest["sft_passed"] = sft_passed
    manifest["sft_sample_count"] = sft_sample_count
    manifest["sft_max_steps"] = sft_max_steps
    manifest["approval_file"] = approval_file
    manifest["sft_config"] = sft_config
    manifest["sft_temp_dataset"] = str(SFT_TEMP_DATASET)
    manifest["sft_temp_config"] = str(SFT_TEMP_CONFIG)
    manifest["sft_output_dir"] = str(SFT_OUTPUT_DIR)
    manifest["sft_train_dataset"] = sft_train_dataset
    manifest["sft_eval_dataset"] = sft_eval_dataset
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDry-run manifest updated (SFT) -> {MANIFEST_PATH}")
    return manifest


def _update_dpo_manifest(
    dpo_passed: bool,
    dpo_pair_count: int,
    dpo_max_steps: int,
    approval_file: str,
    dpo_config: str,
    dpo_train_dataset: str,
    dpo_eval_dataset: str,
) -> dict:
    """Update DPO fields in the dry-run manifest, preserving SFT fields."""
    manifest = _load_existing_manifest()
    manifest["dpo_passed"] = dpo_passed
    manifest["dpo_pair_count"] = dpo_pair_count
    manifest["dpo_max_steps"] = dpo_max_steps
    manifest["approval_file"] = approval_file
    manifest["dpo_config"] = dpo_config
    manifest["dpo_temp_dataset"] = str(DPO_TEMP_DATASET)
    manifest["dpo_temp_config"] = str(DPO_TEMP_CONFIG)
    manifest["dpo_output_dir"] = str(DPO_OUTPUT_DIR)
    manifest["dpo_train_dataset"] = dpo_train_dataset
    manifest["dpo_eval_dataset"] = dpo_eval_dataset
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDry-run manifest updated (DPO) -> {MANIFEST_PATH}")
    return manifest


def _run_sft_dryrun(config_path: Path, spec, approval_file: str, sample_count: int, max_steps: int) -> None:
    print(f"\nPreparing SFT dry-run:")
    print(f"  sample_count: {sample_count}")
    print(f"  max_steps: {max_steps}")

    sft_train = spec.train_dataset_path
    if sft_train is None:
        print("ERROR: No train dataset path in config.")
        sys.exit(1)

    _create_sft_dryrun_dataset(sft_train, sample_count)
    _generate_dryrun_config(config_path, SFT_TEMP_CONFIG, SFT_OUTPUT_DIR, "sft")
    exit_code, stdout, stderr = _run_llamafactory_train(SFT_TEMP_CONFIG)

    sft_passed = exit_code == 0
    print(f"\nSFT dry-run {'PASSED' if sft_passed else f'FAILED (exit code: {exit_code})'}.")

    _update_sft_manifest(
        sft_passed=sft_passed,
        sft_sample_count=sample_count,
        sft_max_steps=max_steps,
        approval_file=approval_file,
        sft_config=str(config_path),
        sft_train_dataset=str(sft_train),
        sft_eval_dataset=str(spec.eval_dataset_path) if spec.eval_dataset_path else "",
    )
    if not sft_passed:
        sys.exit(exit_code)


def _run_dpo_dryrun(config_path: Path, spec, approval_file: str, sample_count: int, max_steps: int) -> None:
    print(f"\nPreparing DPO dry-run:")
    print(f"  pair_count: {sample_count}")
    print(f"  max_steps: {max_steps}")

    dpo_train = spec.train_dataset_path
    if dpo_train is None:
        print("ERROR: No train dataset path in config.")
        sys.exit(1)

    _create_dpo_dryrun_dataset(dpo_train, sample_count)
    _generate_dryrun_config(
        config_path, DPO_TEMP_CONFIG, DPO_OUTPUT_DIR, "dpo",
        extra_overrides={"pref_beta": 0.1, "pref_loss": "sigmoid"},
    )
    exit_code, stdout, stderr = _run_llamafactory_train(DPO_TEMP_CONFIG)

    dpo_passed = exit_code == 0
    print(f"\nDPO dry-run {'PASSED' if dpo_passed else f'FAILED (exit code: {exit_code})'}.")

    _update_dpo_manifest(
        dpo_passed=dpo_passed,
        dpo_pair_count=sample_count,
        dpo_max_steps=max_steps,
        approval_file=approval_file,
        dpo_config=str(config_path),
        dpo_train_dataset=str(dpo_train),
        dpo_eval_dataset=str(spec.eval_dataset_path) if spec.eval_dataset_path else "",
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

    if not args.approval_file:
        print("ERROR: --approval-file is required. Checkpoint B is not approved.")
        sys.exit(1)

    approval = _validate_approval(Path(args.approval_file))

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

    if args.stage == "dpo":
        _run_dpo_dryrun(config_path, spec, args.approval_file, args.sample_count, args.max_steps)
    else:
        _run_sft_dryrun(config_path, spec, args.approval_file, args.sample_count, args.max_steps)


if __name__ == "__main__":
    main()
