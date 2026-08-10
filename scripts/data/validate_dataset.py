"""Phase 03 dataset validation CLI.

Usage:
  uv run python scripts/data/validate_dataset.py --raw data/raw/phase03_smoke_v0.1.jsonl --frozen data/eval/frozen_test.jsonl --diagnostic data/dev/diagnostic_dev.jsonl --manifest-out data/processed/phase03_smoke_manifest.v0.1.json --data-card-out data/processed/phase03_smoke_data_card.v0.1.json
"""

import argparse
from pathlib import Path

from homechef_booking.data.contamination import check_raw_contamination
from homechef_booking.data.manifest import write_dataset_data_card, write_dataset_manifest
from homechef_booking.data.raw_validator import validate_raw_jsonl


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
        # Check each eval suite separately for accurate overlap counts
        if frozen_path:
            frozen_report = check_raw_contamination(raw_path, [frozen_path])
            frozen_overlap = frozen_report.overlap_count
        if diagnostic_path:
            diag_report = check_raw_contamination(raw_path, [diagnostic_path])
            diagnostic_overlap = diag_report.overlap_count

    write_dataset_manifest(
        output_path=Path(args.manifest_out),
        dataset_version="phase03_v0.1",
        raw_path=raw_path,
        sft_train_path=Path(args.sft_train) if args.sft_train else None,
        sft_val_path=Path(args.sft_val) if args.sft_val else None,
        dpo_train_path=Path(args.dpo_train) if args.dpo_train else None,
        dpo_val_path=Path(args.dpo_val) if args.dpo_val else None,
        frozen_eval_overlap=frozen_overlap,
        diagnostic_dev_overlap=diagnostic_overlap,
        generator="deterministic_smoke",
    )
    write_dataset_data_card(
        output_path=Path(args.data_card_out),
        dataset_version="phase03_v0.1",
        frozen_eval_overlap=frozen_overlap,
        diagnostic_dev_overlap=diagnostic_overlap,
    )
    print(f"Manifest written to {args.manifest_out}")
    print(f"Data card written to {args.data_card_out}")
    print("Validation PASSED")


if __name__ == "__main__":
    main()
