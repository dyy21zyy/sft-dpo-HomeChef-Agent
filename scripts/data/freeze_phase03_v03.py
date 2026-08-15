"""Phase 03 → Phase 04 Handoff — Freeze Phase03 v0.3 data + verify SHA256."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

FILES = {
    "RAW": "data/raw/phase03_raw_strong_model_600_v0.3.jsonl",
    "SFT_TRAIN": "data/processed/sft/v0.3/train.jsonl",
    "SFT_VAL": "data/processed/sft/v0.3/val.jsonl",
    "DPO_TRAIN": "data/processed/dpo/v0.3/train.jsonl",
    "DPO_VAL": "data/processed/dpo/v0.3/val.jsonl",
}

OUT = Path("reports/generated/phase03/v0.3/phase03_v0.3_freeze.json")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    result = {"version": "phase03_v0.3", "frozen": True}
    ok = True
    for key, rel in FILES.items():
        p = Path(rel)
        if not p.exists():
            result[key] = {"status": "MISSING", "path": rel}
            ok = False
            continue
        result[key] = {"path": rel, "sha256": sha(p)}
        # sanity count
        n = len(p.read_text(encoding="utf-8").splitlines())
        result[key]["lines"] = n
    result["frozen"] = ok
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
