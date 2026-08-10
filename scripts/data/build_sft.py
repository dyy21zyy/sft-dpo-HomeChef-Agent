"""Phase 03 SFT dataset builder CLI.

Usage:
  uv run python scripts/data/build_sft.py --raw data/raw/phase03_smoke_v0.1.jsonl --train data/processed/phase03_smoke_sft_train.jsonl --val data/processed/phase03_smoke_sft_val.jsonl --val-ratio 0.20 --seed 3001
"""

import argparse
from pathlib import Path

from homechef_booking.data.sft_render import build_sft_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT dataset from validated raw samples")
    parser.add_argument("--raw", required=True, type=str, help="Path to raw JSONL file")
    parser.add_argument("--train", required=True, type=str, help="Path to output SFT train JSONL")
    parser.add_argument("--val", required=True, type=str, help="Path to output SFT val JSONL")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=3001, help="Random seed for split")
    args = parser.parse_args()

    samples = build_sft_dataset(
        raw_path=Path(args.raw),
        train_path=Path(args.train),
        val_path=Path(args.val),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"SFT dataset built: {len(samples)} total samples from {args.raw}")
    print(f"  Train: {args.train}")
    print(f"  Val:   {args.val}")


if __name__ == "__main__":
    main()
