"""Phase 03 → Phase 04 Handoff — Dataset Load Smoke (no training).

Loads the frozen v0.3 SFT/DPO via the project's real validators/loaders and
checks:
  - SFT = 540 train / 60 val
  - DPO = 216 train / 24 val
  - SFT assistant target parseable
  - DPO prompt/chosen/rejected in LLaMA-Factory-compatible form
  - Random 10-sample spot check per split: no empty assistant, no empty
    chosen/rejected, chosen != rejected, valid role, valid JSON, correct
    source_raw_id.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

from homechef_booking.training.config import DatasetRegistry
from homechef_booking.training.dataset_adapter import (
    validate_dpo_for_training,
    validate_sft_for_training,
)

SEED = 42
SPOT = 10

SFT_TRAIN = Path("data/processed/sft/v0.3/train.jsonl")
SFT_VAL = Path("data/processed/sft/v0.3/val.jsonl")
DPO_TRAIN = Path("data/processed/dpo/v0.3/train.jsonl")
DPO_VAL = Path("data/processed/dpo/v0.3/val.jsonl")

VALID_ROLES = {"user", "assistant", "system", "tool"}


def _load_rows(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _spot_check_sft(rows: list[dict], rng: random.Random) -> list[str]:
    errors = []
    sample = rng.sample(rows, min(SPOT, len(rows)))
    for row in sample:
        msgs = row.get("messages", [])
        if not msgs:
            errors.append(f"{row['id']}: empty messages")
            continue
        for m in msgs:
            if m.get("role") not in VALID_ROLES:
                errors.append(f"{row['id']}: invalid role {m.get('role')}")
        # The FINAL assistant message is the real assistant target and must be
        # non-empty (the decision JSON). Intermediate assistant messages may be
        # empty content IF they are tool-call turns (followed by a tool result),
        # which is the legitimate tool-calling conversation format.
        asst = [m for m in msgs if m.get("role") == "assistant"]
        if not asst:
            errors.append(f"{row['id']}: no assistant message")
            continue
        if not asst[-1].get("content"):
            errors.append(f"{row['id']}: empty final assistant target")
        # Any intermediate empty assistant must be a tool-call turn (i.e. a tool
        # message follows it).
        for i, m in enumerate(msgs):
            if m.get("role") == "assistant" and not m.get("content"):
                # Must be followed by a tool message (or be the last).
                if i + 1 < len(msgs) and msgs[i + 1].get("role") != "tool":
                    errors.append(f"{row['id']}: empty assistant not followed by tool result")
        # raw_id must be present and non-empty.
        if not row.get("raw_id"):
            errors.append(f"{row['id']}: missing source_raw_id")
    return errors


def _spot_check_dpo(rows: list[dict], rng: random.Random) -> list[str]:
    errors = []
    sample = rng.sample(rows, min(SPOT, len(rows)))
    for row in sample:
        prompt = row.get("prompt", [])
        chosen = row.get("chosen", "")
        rejected = row.get("rejected", "")
        if not prompt:
            errors.append(f"{row['id']}: empty prompt")
        for m in prompt:
            if m.get("role") not in VALID_ROLES:
                errors.append(f"{row['id']}: invalid prompt role {m.get('role')}")
        if not chosen:
            errors.append(f"{row['id']}: empty chosen")
        if not rejected:
            errors.append(f"{row['id']}: empty rejected")
        if chosen == rejected:
            errors.append(f"{row['id']}: chosen == rejected")
        # chosen/rejected must be valid decision JSON.
        for field in ("chosen", "rejected"):
            try:
                json.loads(row[field])
            except json.JSONDecodeError:
                errors.append(f"{row['id']}: {field} invalid JSON")
        if not row.get("raw_id"):
            errors.append(f"{row['id']}: missing source_raw_id")
    return errors


def main() -> int:
    rng = random.Random(SEED)
    print("=" * 60)
    print("DATASET LOAD SMOKE (v0.3)")
    print("=" * 60)

    # 1. SFT counts via real validator.
    sft_train = validate_sft_for_training(SFT_TRAIN, 540)
    sft_val = validate_sft_for_training(SFT_VAL, 60)
    print(f"SFT train: total={sft_train.total_rows} passed={sft_train.passed} errors={sft_train.error_count}")
    print(f"SFT val:   total={sft_val.total_rows} passed={sft_val.passed} errors={sft_val.error_count}")

    # 2. DPO counts via real validator.
    dpo_train = validate_dpo_for_training(DPO_TRAIN, 216)
    dpo_val = validate_dpo_for_training(DPO_VAL, 24)
    print(f"DPO train: total={dpo_train.total_rows} passed={dpo_train.passed} errors={dpo_train.error_count}")
    print(f"DPO val:   total={dpo_val.total_rows} passed={dpo_val.passed} errors={dpo_val.error_count}")

    # 3. Random spot checks.
    print(f"\nSpot checks (seed={SEED}, n={SPOT}):")
    sft_train_errs = _spot_check_sft(_load_rows(SFT_TRAIN), rng)
    sft_val_errs = _spot_check_sft(_load_rows(SFT_VAL), rng)
    dpo_train_errs = _spot_check_dpo(_load_rows(DPO_TRAIN), rng)
    dpo_val_errs = _spot_check_dpo(_load_rows(DPO_VAL), rng)
    for name, errs in [
        ("SFT train", sft_train_errs), ("SFT val", sft_val_errs),
        ("DPO train", dpo_train_errs), ("DPO val", dpo_val_errs),
    ]:
        print(f"  {name}: {'PASS' if not errs else 'FAIL'} ({len(errs)} issues)")
        for e in errs[:5]:
            print(f"    - {e}")

    # 4. LLaMA-Factory dataset registry validates.
    reg = DatasetRegistry.load(Path("configs/training/phase04_dataset_info.json"))
    reg_errors = reg.validate()
    print(f"\nLLaMA-Factory registry: {'PASS' if not reg_errors else 'FAIL'}")
    for e in reg_errors:
        print(f"  - {e}")

    all_ok = (
        sft_train.passed and sft_val.passed and dpo_train.passed and dpo_val.passed
        and not sft_train_errs and not sft_val_errs
        and not dpo_train_errs and not dpo_val_errs
        and not reg_errors
    )
    print(f"\nSFT_LOAD_SMOKE: {'PASS' if (sft_train.passed and sft_val.passed and not sft_train_errs and not sft_val_errs) else 'FAIL'}")
    print(f"DPO_LOAD_SMOKE: {'PASS' if (dpo_train.passed and dpo_val.passed and not dpo_train_errs and not dpo_val_errs) else 'FAIL'}")
    print(f"DATASET_LOAD_SMOKE: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
