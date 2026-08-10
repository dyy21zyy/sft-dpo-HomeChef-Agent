"""Phase 03 dataset validation CLI.

Usage:
  uv run python scripts/data/validate_dataset.py --raw data/raw/phase03_smoke_v0.1.jsonl --sft-train data/processed/phase03_smoke_sft_train.jsonl --sft-val data/processed/phase03_smoke_sft_val.jsonl --dpo-train data/processed/phase03_smoke_dpo_train.jsonl --dpo-val data/processed/phase03_smoke_dpo_val.jsonl --frozen data/eval/frozen_test.jsonl --diagnostic data/dev/diagnostic_dev.jsonl --manifest-out data/processed/phase03_smoke_manifest.v0.1.json --data-card-out data/processed/phase03_smoke_data_card.v0.1.json
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from homechef_booking.data.contamination import check_raw_contamination
from homechef_booking.data.manifest import write_dataset_data_card, write_dataset_manifest
from homechef_booking.data.raw_validator import validate_raw_jsonl


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _count_duplicate_hashes(hashes: list[str]) -> int:
    """Return the number of entries that are duplicates (total - unique)."""
    counts = Counter(hashes)
    return sum(count for count in counts.values() if count > 1) - sum(1 for count in counts.values() if count > 1)


def _train_val_overlap(train_hashes: list[str], val_hashes: list[str]) -> int:
    """Return the number of hashes that appear in both train and val."""
    return len(set(train_hashes) & set(val_hashes))


def compute_raw_input_fingerprint_duplicates(raw_path: Path) -> int:
    """Count duplicate raw input fingerprints (hash of user_input + current_state)."""
    hashes = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        inp = row.get("input", {})
        fingerprint_data = json.dumps({
            "user_input": inp.get("user_input", ""),
            "current_state": inp.get("current_state", {}),
            "history": inp.get("history", []),
        }, ensure_ascii=False, sort_keys=True)
        hashes.append(hashlib.sha256(fingerprint_data.encode()).hexdigest())
    return _count_duplicate_hashes(hashes)


def compute_sft_duplicates(sft_path: Path | None) -> int:
    """Count duplicate SFT prompt+completion hash pairs."""
    if sft_path is None or not sft_path.exists():
        return 0
    hashes = []
    for line in sft_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        h = f"{row.get('prompt_sha256', '')}:{row.get('completion_sha256', '')}"
        hashes.append(h)
    return _count_duplicate_hashes(hashes)


def compute_dpo_duplicates(dpo_path: Path | None) -> int:
    """Count duplicate DPO pairs (by chosen_sha256 + rejected_sha256 + prompt hash)."""
    if dpo_path is None or not dpo_path.exists():
        return 0
    hashes = []
    for line in dpo_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        prompt_hash = hashlib.sha256(
            json.dumps(row.get("prompt", []), ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        h = f"{row.get('chosen_sha256', '')}:{row.get('rejected_sha256', '')}:{prompt_hash}"
        hashes.append(h)
    return _count_duplicate_hashes(hashes)


def compute_sft_train_val_overlap(train_path: Path | None, val_path: Path | None) -> int:
    """Count SFT hashes that appear in both train and val."""
    if train_path is None or val_path is None or not train_path.exists() or not val_path.exists():
        return 0
    train_hashes = []
    val_hashes = []
    for path, target in [(train_path, train_hashes), (val_path, val_hashes)]:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            h = f"{row.get('prompt_sha256', '')}:{row.get('completion_sha256', '')}"
            target.append(h)
    return _train_val_overlap(train_hashes, val_hashes)


def compute_dpo_train_val_overlap(train_path: Path | None, val_path: Path | None) -> int:
    """Count DPO hashes that appear in both train and val."""
    if train_path is None or val_path is None or not train_path.exists() or not val_path.exists():
        return 0
    train_hashes = []
    val_hashes = []
    for path, target in [(train_path, train_hashes), (val_path, val_hashes)]:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt_hash = hashlib.sha256(
                json.dumps(row.get("prompt", []), ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            h = f"{row.get('chosen_sha256', '')}:{row.get('rejected_sha256', '')}:{prompt_hash}"
            target.append(h)
    return _train_val_overlap(train_hashes, val_hashes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 03 dataset")
    parser.add_argument("--raw", required=True, type=str, help="Path to raw JSONL")
    parser.add_argument("--sft-train", type=str, default=None)
    parser.add_argument("--sft-val", type=str, default=None)
    parser.add_argument("--dpo-train", type=str, default=None)
    parser.add_argument("--dpo-val", type=str, default=None)
    parser.add_argument("--frozen", type=str, default=None)
    parser.add_argument("--diagnostic", type=str, default=None)
    parser.add_argument("--manifest-out", required=True, type=str)
    parser.add_argument("--data-card-out", required=True, type=str)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    sft_train_path = Path(args.sft_train) if args.sft_train else None
    sft_val_path = Path(args.sft_val) if args.sft_val else None
    dpo_train_path = Path(args.dpo_train) if args.dpo_train else None
    dpo_val_path = Path(args.dpo_val) if args.dpo_val else None

    report = validate_raw_jsonl(raw_path)
    print(f"Raw validation: {report.total} rows, {report.valid} valid, {report.error_count} errors")
    if report.error_count:
        for err in report.errors:
            print(f"  [{err.row}] {err.sample_id}: {err.path} — {err.message}")
        raise SystemExit(1)

    eval_paths = []
    frozen_path = None
    diagnostic_path = None
    if args.frozen:
        frozen_path = Path(args.frozen)
        eval_paths.append(frozen_path)
    if args.diagnostic:
        diagnostic_path = Path(args.diagnostic)
        eval_paths.append(diagnostic_path)

    frozen_overlap = 0
    diagnostic_overlap = 0
    if eval_paths:
        contamination = check_raw_contamination(raw_path, eval_paths)
        print(f"Contamination check: {contamination.overlap_count} overlaps with eval suites")
        if contamination.overlapping_ids:
            print(f"  Overlapping IDs: {contamination.overlapping_ids}")
        if frozen_path:
            frozen_report = check_raw_contamination(raw_path, [frozen_path])
            frozen_overlap = frozen_report.overlap_count
        if diagnostic_path:
            diag_report = check_raw_contamination(raw_path, [diagnostic_path])
            diagnostic_overlap = diag_report.overlap_count

    # Duplicate protection metrics
    raw_dup = compute_raw_input_fingerprint_duplicates(raw_path)
    sft_dup = compute_sft_duplicates(sft_train_path) + compute_sft_duplicates(sft_val_path)
    dpo_dup = compute_dpo_duplicates(dpo_train_path) + compute_dpo_duplicates(dpo_val_path)
    sft_overlap = compute_sft_train_val_overlap(sft_train_path, sft_val_path)
    dpo_overlap = compute_dpo_train_val_overlap(dpo_train_path, dpo_val_path)

    print(f"raw_input_fingerprint_duplicate_count: {raw_dup}")
    print(f"sft_prompt_completion_duplicate_count: {sft_dup}")
    print(f"dpo_pair_duplicate_count: {dpo_dup}")
    print(f"sft_train_val_overlap_by_hash: {sft_overlap}")
    print(f"dpo_train_val_overlap_by_hash: {dpo_overlap}")

    write_dataset_manifest(
        output_path=Path(args.manifest_out),
        dataset_version="phase03_v0.1",
        raw_path=raw_path,
        sft_train_path=sft_train_path,
        sft_val_path=sft_val_path,
        dpo_train_path=dpo_train_path,
        dpo_val_path=dpo_val_path,
        frozen_eval_overlap=frozen_overlap,
        diagnostic_dev_overlap=diagnostic_overlap,
        generator="deterministic_smoke",
        raw_input_fingerprint_duplicate_count=raw_dup,
        sft_prompt_completion_duplicate_count=sft_dup,
        dpo_pair_duplicate_count=dpo_dup,
        sft_train_val_overlap_by_hash=sft_overlap,
        dpo_train_val_overlap_by_hash=dpo_overlap,
    )
    write_dataset_data_card(
        output_path=Path(args.data_card_out),
        dataset_version="phase03_v0.1",
        frozen_eval_overlap=frozen_overlap,
        diagnostic_dev_overlap=diagnostic_overlap,
        raw_input_fingerprint_duplicate_count=raw_dup,
        sft_prompt_completion_duplicate_count=sft_dup,
        dpo_pair_duplicate_count=dpo_dup,
        sft_train_val_overlap_by_hash=sft_overlap,
        dpo_train_val_overlap_by_hash=dpo_overlap,
    )
    print(f"Manifest written to {args.manifest_out}")
    print(f"Data card written to {args.data_card_out}")
    print("Validation PASSED")


if __name__ == "__main__":
    main()
