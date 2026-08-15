"""Phase 04 Task 2 — Render a resolved LLaMA-Factory training config.

Validates a training YAML config, validates the referenced dataset files, maps
project dataset paths to the LLaMA-Factory data/dataset_info.json registry names
(dataset / eval_dataset), strips project-only metadata, and writes a resolved
copy that is directly consumable by llamafactory-cli.

Usage:
  uv run python scripts/train/render_config.py --config configs/training/phase04_sft_qwen3_1_7b.yaml --output experiments/phase04/configs/phase04_sft_qwen3_1_7b.resolved.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from homechef_booking.training.config import load_training_run_spec, validate_training_run_spec
from homechef_booking.training.dataset_adapter import (
    validate_dpo_for_training,
    validate_sft_for_training,
)
from homechef_booking.training.renderer import (
    assert_rendered_config,
    render_training_spec,
)


def _validate_datasets(spec) -> None:
    """Validate the dataset files referenced by the spec (no Phase03 mutation)."""
    if spec.train_dataset_path:
        report = (
            validate_dpo_for_training(spec.train_dataset_path, expected_count=216)
            if spec.stage == "dpo"
            else validate_sft_for_training(spec.train_dataset_path, expected_count=540)
        )
        if not report.passed:
            for err in report.errors:
                print(f"  - {err}")
            raise SystemExit(1)
        print(f"Train dataset OK: {report.total_rows} rows")
    if spec.eval_dataset_path:
        report = (
            validate_dpo_for_training(spec.eval_dataset_path, expected_count=24)
            if spec.stage == "dpo"
            else validate_sft_for_training(spec.eval_dataset_path, expected_count=60)
        )
        if not report.passed:
            for err in report.errors:
                print(f"  - {err}")
            raise SystemExit(1)
        print(f"Eval dataset OK: {report.total_rows} rows")


def render_training_config(
    config_path: Path,
    output_path: Path,
    *,
    dataset_info_path: Path = Path("data/dataset_info.json"),
) -> dict[str, Any]:
    """Load, validate, render (dataset mapping + metadata filter), and gate.

    Returns the resolved LLaMA-Factory-native config dict.
    """
    spec = load_training_run_spec(config_path)
    config_errors = validate_training_run_spec(spec)
    if config_errors:
        print(f"Config validation FAILED for {config_path}:")
        for err in config_errors:
            print(f"  - {err}")
        raise SystemExit(1)

    _validate_datasets(spec)

    resolved = render_training_spec(spec, dataset_info_path=dataset_info_path)

    # Fail-closed gate before writing / launching.
    assert_rendered_config(resolved, stage=spec.stage)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(resolved, default_flow_style=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"Resolved config written to {output_path}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a resolved training config")
    parser.add_argument("--config", required=True, type=str, help="Path to training YAML config")
    parser.add_argument("--output", required=True, type=str, help="Output path for resolved YAML")
    args = parser.parse_args()

    render_training_config(Path(args.config), Path(args.output))


if __name__ == "__main__":
    main()
