"""Phase 03 dataset manifest and data card writers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from homechef_booking.data.dataset_schema import DatasetDataCard, DatasetManifest
from homechef_booking.evaluation.suite_manifest import compute_sha256


def write_dataset_manifest(
    output_path: Path,
    dataset_version: str,
    raw_path: Path,
    sft_train_path: Path | None = None,
    sft_val_path: Path | None = None,
    dpo_train_path: Path | None = None,
    dpo_val_path: Path | None = None,
    frozen_eval_overlap: int = 0,
    diagnostic_dev_overlap: int = 0,
    seed: int = 3001,
) -> Path:
    raw_count = len([l for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()])
    manifest = DatasetManifest(
        dataset_version=dataset_version,
        created_at=datetime.now(timezone.utc).isoformat(),
        raw_path=str(raw_path),
        raw_sha256=compute_sha256(raw_path),
        raw_count=raw_count,
        sft_train_path=str(sft_train_path) if sft_train_path else None,
        sft_train_sha256=compute_sha256(sft_train_path) if sft_train_path and sft_train_path.exists() else None,
        sft_train_count=_count_lines(sft_train_path),
        sft_val_path=str(sft_val_path) if sft_val_path else None,
        sft_val_sha256=compute_sha256(sft_val_path) if sft_val_path and sft_val_path.exists() else None,
        sft_val_count=_count_lines(sft_val_path),
        dpo_train_path=str(dpo_train_path) if dpo_train_path else None,
        dpo_train_sha256=compute_sha256(dpo_train_path) if dpo_train_path and dpo_train_path.exists() else None,
        dpo_train_count=_count_lines(dpo_train_path),
        dpo_val_path=str(dpo_val_path) if dpo_val_path else None,
        dpo_val_sha256=compute_sha256(dpo_val_path) if dpo_val_path and dpo_val_path.exists() else None,
        dpo_val_count=_count_lines(dpo_val_path),
        frozen_eval_overlap=frozen_eval_overlap,
        diagnostic_dev_overlap=diagnostic_dev_overlap,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest.model_dump(mode="json", exclude_none=False), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def write_dataset_data_card(
    output_path: Path,
    dataset_version: str,
    frozen_eval_overlap: int = 0,
    diagnostic_dev_overlap: int = 0,
    phase02_dependency_status: str = "not_used_for_training",
) -> Path:
    card = DatasetDataCard(
        dataset_version=dataset_version,
        created_at=datetime.now(timezone.utc).isoformat(),
        frozen_eval_overlap=frozen_eval_overlap,
        diagnostic_dev_overlap=diagnostic_dev_overlap,
        phase02_base_benchmark_dependency=phase02_dependency_status,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(card.model_dump(mode="json", exclude_none=False), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _count_lines(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return len([l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()])
