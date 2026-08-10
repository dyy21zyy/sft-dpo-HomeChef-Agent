"""Phase 04 Task 3 — Package training artifacts for cloud deployment.

Packages training config, dataset registry, dataset manifests, and a generated
README. Refuses to proceed without --approval-file.

If requirements-train.txt is missing (Checkpoint A not approved), fails closed
or reports approval_required. Does NOT silently create requirements-train.txt.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from homechef_booking.training.config import load_training_run_spec, validate_training_run_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Package cloud training artifacts")
    parser.add_argument("--config", required=True, type=str, help="Path to training YAML config")
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory for package")
    parser.add_argument("--approval-file", type=str, default=None,
                        help="Path to approval JSON (required)")
    args = parser.parse_args()

    # Refuse without approval
    if not args.approval_file:
        print("ERROR: --approval-file is required.")
        print("  Package cannot be created without approval.")
        sys.exit(1)

    approval_path = Path(args.approval_file)
    if not approval_path.exists():
        print(f"ERROR: Approval file not found: {args.approval_file}")
        sys.exit(1)

    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        if not approval.get("approved"):
            print("ERROR: Approval file exists but 'approved' is not true.")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect artifacts
    artifacts: dict[str, str] = {}
    missing: list[str] = []

    # 1. Training YAML
    if config_path.exists():
        artifacts["training_config"] = str(config_path)

    # 2. Dataset registry
    registry_path = Path("configs/training/phase04_dataset_info.json")
    if registry_path.exists():
        artifacts["dataset_registry"] = str(registry_path)
    else:
        missing.append(str(registry_path))

    # 3. Phase 03 dataset manifest
    ds_manifest = Path("data/processed/phase03_dataset_manifest.v0.1.json")
    if ds_manifest.exists():
        artifacts["dataset_manifest"] = str(ds_manifest)
    else:
        missing.append(str(ds_manifest))

    # 4. Phase 03 data card
    ds_card = Path("data/processed/phase03_data_card.v0.1.json")
    if ds_card.exists():
        artifacts["data_card"] = str(ds_card)
    else:
        missing.append(str(ds_card))

    # 5. Contract manifest (Phase 00)
    contract = Path("data/processed/phase03_dataset_manifest.v0.1.json")
    if contract.exists():
        artifacts["contract_manifest"] = str(contract)

    # 6. requirements-train.txt (Checkpoint A gate)
    req_path = Path("requirements-train.txt")
    if req_path.exists():
        artifacts["requirements"] = str(req_path)
    else:
        missing.append("requirements-train.txt")
        print("WARNING: requirements-train.txt not found.")
        print("  Checkpoint A is NOT approved. This file is not created.")
        print("  approval_required: requirements-train.txt must be created after Checkpoint A approval.")

    # 7. Exact dataset paths from config
    if spec.train_dataset_path and spec.train_dataset_path.exists():
        artifacts["train_dataset"] = str(spec.train_dataset_path)
    else:
        missing.append(str(spec.train_dataset_path))

    if spec.eval_dataset_path and spec.eval_dataset_path.exists():
        artifacts["eval_dataset"] = str(spec.eval_dataset_path)
    else:
        missing.append(str(spec.eval_dataset_path))

    # 8. Generated README
    readme_lines = [
        f"# Phase 04 Training Package",
        f"",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"",
        f"## Config",
        f"- Stage: {spec.stage}",
        f"- Model: {spec.model_name_or_path}",
        f"- Training config: {args.config}",
        f"",
        f"## Artifacts",
    ]
    for name, path in sorted(artifacts.items()):
        readme_lines.append(f"- {name}: {path}")
    if missing:
        readme_lines.append(f"")
        readme_lines.append(f"## Missing")
        for m in missing:
            readme_lines.append(f"- {m} (approval_required)")
    readme_lines.append(f"")
    readme_lines.append(f"## Approval")
    readme_lines.append(f"- Approval file: {args.approval_file}")

    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

    # 9. Package manifest
    package_manifest = {
        "package_type": "phase04_training_artifacts",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "artifacts": artifacts,
        "missing": missing,
        "approval_file": args.approval_file,
        "checkpoint_a_approved": "requirements-train.txt" not in missing,
    }
    manifest_path = output_dir / "package_manifest.json"
    manifest_path.write_text(json.dumps(package_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Package created at {output_dir}")
    print(f"  Artifacts: {len(artifacts)} collected")
    print(f"  Missing: {len(missing)} (approval_required)")
    if missing:
        print(f"  WARNING: Some artifacts require Checkpoint A approval.")
        print(f"  Do NOT deploy without all artifacts.")


if __name__ == "__main__":
    main()
