"""Phase 03 raw dataset generator CLI.

Smoke mode:
  uv run python scripts/data/generate_raw.py --mode smoke --count 25 --seed 3001 --output data/raw/phase03_smoke_v0.1.jsonl

Full mode (requires ChatGPT approval):
  uv run python scripts/data/generate_raw.py --mode full --count 600 --seed 3001 --approved-schema-file project-log/phase03_schema_approval.json --output data/raw/phase03_raw_v0.1.jsonl
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 03 raw dataset")
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=3001)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--approved-schema-file", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "full" and not args.approved_schema_file:
        print("ERROR: --mode full requires --approved-schema-file (ChatGPT approval file)")
        raise SystemExit(1)
    if args.mode == "full":
        approval_path = Path(args.approved_schema_file)
        if not approval_path.exists():
            print(f"ERROR: Approval file not found: {args.approved_schema_file}")
            raise SystemExit(1)
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        if not approval.get("approved"):
            print("ERROR: Approval file exists but 'approved' field is not true")
            raise SystemExit(1)

    from homechef_booking.data.generator import generate_smoke_raw
    output_path = Path(args.output)
    result = generate_smoke_raw(output_path, count=args.count, seed=args.seed)
    print(f"Generated {args.count} raw samples to {result}")


if __name__ == "__main__":
    main()
