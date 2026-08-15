"""Phase 03 v0.3 — Deterministic SFT derivation from the validated strong-model Raw.

Derives SFT train (540) / val (60) from the SAME validated 600 Raw. No model is
called; the assistant target comes directly from Raw.expected via the real
PromptBuilder renderer. Deterministic (fixed seed), 9:1, stratified by
difficulty, zero train/val source_raw_id leakage.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

from homechef_booking.data.sft_render import render_sft_sample

RAW_PATH = Path("data/raw/phase03_raw_strong_model_600_v0.3.jsonl")
OUT_DIR = Path("data/processed/sft/v0.3")
REPORT_DIR = Path("reports/generated/phase03/v0.3")

SEED = 3001
TRAIN_COUNT = 540
VAL_COUNT = 60


def _difficulty_of(sample) -> str:
    tags = set(sample.tags)
    for lvl in ("easy", "medium", "hard"):
        if f"difficulty_{lvl}" in tags:
            return lvl
    return sample.generation.difficulty


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    from homechef_booking.data.raw_sample import parse_raw_sample_line
    lines = RAW_PATH.read_text(encoding="utf-8").splitlines()
    raw_samples = [parse_raw_sample_line(line) for line in lines if line.strip()]
    assert len(raw_samples) == 600

    sft = [render_sft_sample(r) for r in raw_samples]

    # Deterministic stratified split by difficulty (9:1).
    rng = random.Random(SEED)
    by_diff = {"easy": [], "medium": [], "hard": []}
    for s in sft:
        by_diff[_difficulty_of(_raw_for_sft(s, raw_samples))].append(s)

    train = []
    val = []
    for lvl, pool in by_diff.items():
        rng.shuffle(pool)
        val_target = max(1, round(len(pool) * 0.1))
        val.extend(pool[:val_target])
        train.extend(pool[val_target:])

    rng.shuffle(train)
    rng.shuffle(val)

    # Verify counts.
    assert len(train) == TRAIN_COUNT, f"train {len(train)} != 540"
    assert len(val) == VAL_COUNT, f"val {len(val)} != 60"

    # Verify zero source leakage.
    train_raw = {s.raw_id for s in train}
    val_raw = {s.raw_id for s in val}
    leakage = train_raw & val_raw
    assert len(leakage) == 0, f"source leakage: {leakage}"

    # Write.
    train_path = OUT_DIR / "train.jsonl"
    val_path = OUT_DIR / "val.jsonl"
    train_path.write_text("\n".join(json.dumps(s.model_dump(mode="json", exclude_none=False), ensure_ascii=False, sort_keys=True) for s in train), encoding="utf-8")
    val_path.write_text("\n".join(json.dumps(s.model_dump(mode="json", exclude_none=False), ensure_ascii=False, sort_keys=True) for s in val), encoding="utf-8")

    # Split report.
    report = {
        "raw_total": 600,
        "train": len(train),
        "val": len(val),
        "seed": SEED,
        "source_leakage": len(leakage),
        "train_difficulty": dict(Counter(_difficulty_of(_raw_for_sft(s, raw_samples)) for s in train)),
        "val_difficulty": dict(Counter(_difficulty_of(_raw_for_sft(s, raw_samples)) for s in val)),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "deterministic": True,
    }
    (REPORT_DIR / "sft_split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"SFT train={len(train)} val={len(val)} leakage={len(leakage)}")
    print(f"train_difficulty={report['train_difficulty']}")
    print(f"val_difficulty={report['val_difficulty']}")
    return 0


def _raw_for_sft(s, raw_samples):
    return next((r for r in raw_samples if r.id == s.raw_id), None)


if __name__ == "__main__":
    sys.exit(main())
