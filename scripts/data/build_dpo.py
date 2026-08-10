"""Phase 03 DPO dataset builder CLI.

Dense mode (default):
  uv run python scripts/data/build_dpo.py --raw data/raw/phase03_raw_v0.1.jsonl --train data/processed/phase03_dpo_v0.1_train.jsonl --val data/processed/phase03_dpo_v0.1_val.jsonl --val-ratio 0.10 --seed 3001

Targeted mode:
  uv run python scripts/data/build_dpo.py --raw data/raw/phase03_raw_v0.1.jsonl --train data/processed/phase03_dpo_targeted_v0.1_train.jsonl --val data/processed/phase03_dpo_targeted_v0.1_val.jsonl --val-ratio 0.10 --seed 3001 --dpo-policy targeted --targets H1,H3,H4,H5,H6,H7 --target-total-min 180 --max-pairs 240 --min-per-target 25
"""

import argparse
from pathlib import Path

from homechef_booking.data.dpo_pairs import build_dpo_dataset, build_dpo_targeted_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO pairs from validated raw samples")
    parser.add_argument("--raw", required=True, type=str, help="Path to raw JSONL file")
    parser.add_argument("--train", required=True, type=str, help="Path to output DPO train JSONL")
    parser.add_argument("--val", required=True, type=str, help="Path to output DPO val JSONL")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=3001, help="Random seed for split")
    parser.add_argument("--dpo-policy", type=str, choices=["dense", "targeted"], default="dense",
                        help="DPO build policy: dense (all heuristics) or targeted (high-risk only)")
    parser.add_argument("--targets", type=str, default=None,
                        help="Comma-separated list of H targets, e.g. H1,H3,H4,H5,H6,H7")
    parser.add_argument("--min-per-target", type=int, default=25,
                        help="Minimum pairs per target in targeted mode")
    parser.add_argument("--max-pairs", type=int, default=240,
                        help="Maximum total pairs in targeted mode")
    parser.add_argument("--target-total-min", type=int, default=180,
                        help="Minimum total pairs required in targeted mode")
    args = parser.parse_args()

    if args.dpo_policy == "dense":
        pairs = build_dpo_dataset(
            raw_path=Path(args.raw),
            train_path=Path(args.train),
            val_path=Path(args.val),
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        print(f"DPO dataset built (dense): {len(pairs)} total pairs from {args.raw}")
        print(f"  Train: {args.train}")
        print(f"  Val:   {args.val}")
    else:
        if not args.targets:
            print("ERROR: --dpo-policy targeted requires --targets")
            raise SystemExit(1)
        targets = [t.strip() for t in args.targets.split(",")]
        all_pairs, available, distribution = build_dpo_targeted_dataset(
            raw_path=Path(args.raw),
            train_path=Path(args.train),
            val_path=Path(args.val),
            selected_targets=targets,
            min_per_target=args.min_per_target,
            max_pairs=args.max_pairs,
            target_total_min=args.target_total_min,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        print(f"DPO dataset built (targeted): {len(all_pairs)} total pairs from {args.raw}")
        print(f"  Train: {args.train}")
        print(f"  Val:   {args.val}")
        print(f"  Targets selected: {targets}")
        print(f"  Available per target: {available}")
        print(f"  Final distribution: {distribution}")
        min_ok = all(v >= args.min_per_target for v in distribution.values())
        print(f"  min_per_target ({args.min_per_target}) satisfied: {min_ok}")
        if not min_ok:
            short = {k: v for k, v in distribution.items() if v < args.min_per_target}
            print(f"  WARNING: targets below min: {short}")


if __name__ == "__main__":
    main()
