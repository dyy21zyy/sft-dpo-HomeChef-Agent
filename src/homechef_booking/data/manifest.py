"""Phase 03 dataset manifest and data card writers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
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
    generator: str = "openai_responses",
    raw_input_fingerprint_duplicate_count: int = 0,
    sft_prompt_completion_duplicate_count: int = 0,
    dpo_pair_duplicate_count: int = 0,
    sft_train_val_overlap_by_hash: int = 0,
    dpo_train_val_overlap_by_hash: int = 0,
) -> Path:
    raw_count = len([line for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    manifest = DatasetManifest(
        dataset_version=dataset_version,
        created_at=datetime.now(UTC).isoformat(),
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
        raw_input_fingerprint_duplicate_count=raw_input_fingerprint_duplicate_count,
        sft_prompt_completion_duplicate_count=sft_prompt_completion_duplicate_count,
        dpo_pair_duplicate_count=dpo_pair_duplicate_count,
        sft_train_val_overlap_by_hash=sft_train_val_overlap_by_hash,
        dpo_train_val_overlap_by_hash=dpo_train_val_overlap_by_hash,
        seed=seed,
        generator=generator,
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
    raw_input_fingerprint_duplicate_count: int = 0,
    sft_prompt_completion_duplicate_count: int = 0,
    dpo_pair_duplicate_count: int = 0,
    sft_train_val_overlap_by_hash: int = 0,
    dpo_train_val_overlap_by_hash: int = 0,
) -> Path:
    card = DatasetDataCard(
        dataset_version=dataset_version,
        created_at=datetime.now(UTC).isoformat(),
        frozen_eval_overlap=frozen_eval_overlap,
        diagnostic_dev_overlap=diagnostic_dev_overlap,
        phase02_base_benchmark_dependency=phase02_dependency_status,
        raw_input_fingerprint_duplicate_count=raw_input_fingerprint_duplicate_count,
        sft_prompt_completion_duplicate_count=sft_prompt_completion_duplicate_count,
        dpo_pair_duplicate_count=dpo_pair_duplicate_count,
        sft_train_val_overlap_by_hash=sft_train_val_overlap_by_hash,
        dpo_train_val_overlap_by_hash=dpo_train_val_overlap_by_hash,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(card.model_dump(mode="json", exclude_none=False), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _count_lines(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def write_targeted_dpo_manifest(
    output_path: Path,
    dataset_version: str,
    raw_path: Path,
    dpo_train_path: Path,
    dpo_val_path: Path,
    target_distribution: dict[str, int],
    min_per_target: int,
    min_per_target_satisfied: bool,
    total_pairs: int,
    dpo_pair_duplicate_count: int = 0,
    dpo_train_val_overlap_by_hash: int = 0,
    frozen_eval_overlap: int = 0,
    diagnostic_dev_overlap: int = 0,
    seed: int = 3001,
) -> Path:
    """Write a targeted DPO manifest (separate from the dense DPO manifest)."""
    manifest = {
        "manifest_type": "targeted_dpo",
        "dataset_version": dataset_version,
        "contract_id": "homechef-booking-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "dpo_policy": "targeted",
        "dpo_policy_description": "High-risk heuristics only: H1 (fabricated chef), H3 (stale chef ID), H4 (dietary constraint loss), H5 (query dependency mutation), H6 (unauthorized booking confirmation), H7 (candidate order mutation)",
        "dpo_policy_exclusions": "H2 (action type swap — structural, not business safety). tool_error_booking_paused is not covered by current targeted DPO v0.1.",
        "dense_dpo_v0_1_preserved": True,
        "raw_path": str(raw_path),
        "raw_sha256": compute_sha256(raw_path),
        "dpo_train_path": str(dpo_train_path),
        "dpo_train_sha256": compute_sha256(dpo_train_path) if dpo_train_path.exists() else None,
        "dpo_train_count": _count_lines(dpo_train_path),
        "dpo_val_path": str(dpo_val_path),
        "dpo_val_sha256": compute_sha256(dpo_val_path) if dpo_val_path.exists() else None,
        "dpo_val_count": _count_lines(dpo_val_path),
        "total_pairs": total_pairs,
        "target_distribution": target_distribution,
        "min_per_target": min_per_target,
        "min_per_target_satisfied": min_per_target_satisfied,
        "dpo_pair_duplicate_count": dpo_pair_duplicate_count,
        "dpo_train_val_overlap_by_hash": dpo_train_val_overlap_by_hash,
        "frozen_eval_overlap": frozen_eval_overlap,
        "diagnostic_dev_overlap": diagnostic_dev_overlap,
        "seed": seed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def write_targeted_dpo_data_card(
    output_path: Path,
    dataset_version: str,
    target_distribution: dict[str, int],
    total_pairs: int,
    dpo_pair_duplicate_count: int = 0,
    dpo_train_val_overlap_by_hash: int = 0,
    frozen_eval_overlap: int = 0,
    diagnostic_dev_overlap: int = 0,
) -> Path:
    """Write a targeted DPO data card."""
    card = {
        "data_card_type": "targeted_dpo",
        "phase": "03",
        "contract_id": "homechef-booking-v1",
        "dataset_version": dataset_version,
        "created_at": datetime.now(UTC).isoformat(),
        "description": "Targeted DPO is an experimental addition to the dense DPO v0.1 dataset. It covers only high-risk preference boundary corrections.",
        "dpo_not_full_raw_coverage": True,
        "targeted_dpo_purpose": "High-risk preference boundary correction only",
        "dense_dpo_v0_1_preserved": True,
        "total_pairs": total_pairs,
        "target_distribution": target_distribution,
        "train_val_split": "approximately 90/10",
        "total_pairs_controlled": "180-240",
        "no_frozen_test_training_use": True,
        "no_real_user_logs": True,
        "phase02_invalid_output_not_used": True,
        "dpo_pair_duplicate_count": dpo_pair_duplicate_count,
        "dpo_train_val_overlap_by_hash": dpo_train_val_overlap_by_hash,
        "frozen_eval_overlap": frozen_eval_overlap,
        "diagnostic_dev_overlap": diagnostic_dev_overlap,
        "tool_error_booking_paused_coverage": "not covered by current targeted DPO v0.1",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
