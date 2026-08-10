"""Phase 03 DPO dataset builder CLI.

Usage:
  uv run python scripts/data/build_dpo.py --raw data/raw/phase03_smoke_v0.1.jsonl --train data/processed/phase03_smoke_dpo_train.jsonl --val data/processed/phase03_smoke_dpo_val.jsonl --val-ratio 0.20 --seed 3001
"""

import argparse
from pathlib import Path

from homechef_booking.data.dpo_pairs import build_dpo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO pairs from validated raw samples")
    parser.add_argument("--raw", required=True, type=str, help="Path to raw JSONL file")
    parser.add_argument("--train", required=True, type=str, help="Path to output DPO train JSONL")
    parser.add_argument("--val", required=True, type=str, help="Path to output DPO val JSONL")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=3001, help="Random seed for split")
    args = parser.parse_args()

    pairs = build_dpo_dataset(
        raw_path=Path(args.raw),
        train_path=Path(args.train),
        val_path=Path(args.val),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"DPO dataset built: {len(pairs)} total pairs from {args.raw}")
    print(f"  Train: {args.train}")
    print(f"  Val:   {args.val}")


if __name__ == "__main__":
    main()
