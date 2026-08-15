"""Phase 04 Task 3 — Training artifact manifest and dry-run support."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrainingArtifactManifest(BaseModel):
    """Records a training run's dataset provenance, config, and output.

    Required fields per the Phase 04 frozen plan:
    - run_id, stage, model_name_or_path, contract_id
    - dataset_manifest_path, train_dataset_path, eval_dataset_path
    - training_config_path, output_dir
    - max_steps, approval_id, git_sha, created_at
    - hardware, dependency_snapshot_path
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    run_id: str
    stage: str
    model_name_or_path: str
    contract_id: str = "homechef-booking-v1"
    dataset_manifest_path: str
    train_dataset_path: str
    eval_dataset_path: str
    training_config_path: str
    output_dir: str
    max_steps: int = 2
    approval_id: str = ""
    git_sha: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    hardware: dict[str, Any] = Field(default_factory=dict)
    dependency_snapshot_path: str = ""


def write_training_artifact_manifest(manifest: TrainingArtifactManifest, output_path: Path) -> Path:
    """Write a training artifact manifest to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return output_path


def validate_training_artifact_manifest(path: Path) -> list[str]:
    """Validate a training artifact manifest JSON file.

    Returns a list of error messages. Empty list = valid.
    """
    errors: list[str] = []

    if not path.exists():
        errors.append(f"Manifest file not found: {path}")
        return errors

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    required_fields = [
        "run_id",
        "stage",
        "model_name_or_path",
        "contract_id",
        "dataset_manifest_path",
        "train_dataset_path",
        "eval_dataset_path",
        "training_config_path",
        "output_dir",
        "max_steps",
        "approval_id",
        "git_sha",
        "created_at",
        "hardware",
        "dependency_snapshot_path",
    ]

    for field in required_fields:
        if field not in raw:
            errors.append(f"Missing required field: {field}")

    if raw.get("stage") not in ("sft", "dpo", None):
        errors.append(f"Unknown stage: {raw.get('stage')}")

    if raw.get("approval_id") in (None, ""):
        errors.append("approval_id must not be empty")

    if raw.get("contract_id") != "homechef-booking-v1":
        errors.append(f"contract_id must be 'homechef-booking-v1', got: {raw.get('contract_id')}")

    return errors
