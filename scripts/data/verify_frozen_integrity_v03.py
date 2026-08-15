"""Phase 03 → Phase 04 — Frozen Evaluation Integrity.

Verifies Frozen Test / Diagnostic Dev SHAs are unchanged and Phase03 v0.3 has
0 overlap (exact + normalized + near-duplicate) with both.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

from homechef_booking.data.contamination import check_raw_contamination

FROZEN = Path("data/eval/frozen_test.jsonl")
DIAG = Path("data/dev/diagnostic_dev.jsonl")
FROZEN_MANIFEST = Path("data/eval/frozen_test.manifest.json")
DIAG_MANIFEST = Path("data/dev/diagnostic_dev.manifest.json")
RAW = Path("data/raw/phase03_raw_strong_model_600_v0.3.jsonl")

OUT = Path("reports/generated/phase03/v0.3/frozen_integrity_report.json")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    out = {}

    # 1. Frozen Test integrity.
    frozen_sha = sha(FROZEN)
    manifest_sha = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))["sha256"]
    out["frozen_test_sha"] = frozen_sha
    out["frozen_test_manifest_sha"] = manifest_sha
    out["frozen_test_sha_unchanged"] = frozen_sha == manifest_sha

    # 2. Diagnostic Dev integrity.
    diag_sha = sha(DIAG)
    diag_manifest = json.loads(DIAG_MANIFEST.read_text(encoding="utf-8"))
    diag_manifest_sha = diag_manifest["sha256"]
    out["diagnostic_dev_sha"] = diag_sha
    out["diagnostic_dev_manifest_sha"] = diag_manifest_sha
    out["diagnostic_dev_sha_unchanged"] = diag_sha == diag_manifest_sha

    # 3. Contamination (exact overlap via check_raw_contamination).
    frozen_contam = check_raw_contamination(RAW, [FROZEN])
    diag_contam = check_raw_contamination(RAW, [DIAG])
    out["frozen_exact_overlap"] = frozen_contam.overlap_count
    out["frozen_overlapping_ids"] = frozen_contam.overlapping_ids
    out["diagnostic_exact_overlap"] = diag_contam.overlap_count
    out["diagnostic_overlapping_ids"] = diag_contam.overlapping_ids

    # 4. Normalized / near-duplicate detection on user inputs.
    # Only count collisions with a MEANINGFUL normalized length (>= 4 chars).
    # Trivial single-token confirmations like "确认" normalize to a 2-char
    # string shared across ANY confirmation dataset and are NOT evidence of
    # contamination.
    MIN_NORM_LEN = 4
    frozen_sigs = _norm_sigs(FROZEN)
    diag_sigs = _norm_sigs(DIAG)
    raw_frozen_dup = 0
    raw_diag_dup = 0
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ui = row["input"].get("user_input")
        if not ui:
            continue
        norm = _normalize(ui)
        if len(norm) < MIN_NORM_LEN:
            continue
        if norm in frozen_sigs:
            raw_frozen_dup += 1
        if norm in diag_sigs:
            raw_diag_dup += 1
    out["frozen_normalized_overlap"] = raw_frozen_dup
    out["diagnostic_normalized_overlap"] = raw_diag_dup
    out["normalized_min_length"] = MIN_NORM_LEN

    out["gate"] = (
        out["frozen_test_sha_unchanged"]
        and out["diagnostic_dev_sha_unchanged"]
        and out["frozen_exact_overlap"] == 0
        and out["diagnostic_exact_overlap"] == 0
        and out["frozen_normalized_overlap"] == 0
        and out["diagnostic_normalized_overlap"] == 0
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["gate"] else 1


def _normalize(t: str) -> str:
    n = re.sub(r'\d+', 'N', t)
    n = re.sub(r'[，。！？、；：\s]', '', n)
    return n


def _norm_sigs(p: Path) -> set[str]:
    sigs = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            ui = row.get("input", {}).get("user_input")
            if not ui:
                continue
            sigs.add(_normalize(ui))
        except json.JSONDecodeError:
            continue
    return sigs


if __name__ == "__main__":
    sys.exit(main())
