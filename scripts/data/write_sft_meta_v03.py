"""Phase 03 v0.3 — Write SFT manifest + dataset card."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path("data/processed/sft/v0.3")
REPORT_DIR = Path("reports/generated/phase03/v0.3")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    train = OUT_DIR / "train.jsonl"
    val = OUT_DIR / "val.jsonl"

    manifest = {
        "dataset_version": "phase03_v0.3_sft",
        "raw_source": "data/raw/phase03_raw_strong_model_600_v0.3.jsonl",
        "derivation": "deterministic render of validated Raw (no model call)",
        "train_count": len(train.read_text(encoding="utf-8").splitlines()),
        "val_count": len(val.read_text(encoding="utf-8").splitlines()),
        "train_sha256": _sha(train),
        "val_sha256": _sha(val),
        "renderer": "homechef_booking.data.sft_render.render_sft_sample",
        "assistant_target": "Raw.expected (canonical decision JSON)",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_card = (
        "# SFT v0.3 Dataset Card\n\n"
        f"**Version**: phase03_v0.3_sft\n"
        f"**Raw source**: {manifest['raw_source']}\n"
        f"**Derivation**: {manifest['derivation']}\n"
        f"**Train**: {manifest['train_count']} / **Val**: {manifest['val_count']} (9:1 stratified, deterministic seed 3001)\n\n"
        "## Split\n"
        "- train 540 / val 60\n"
        "- source_raw_id zero leakage\n"
        "- difficulty stratified: train 162 easy / 216 medium / 162 hard; val 18/24/18\n\n"
        "## Content\n"
        "- messages from the real PromptBuilder\n"
        "- assistant target directly from Raw.expected (canonical decision JSON)\n"
        "- no model was called to generate SFT\n"
    )
    (OUT_DIR / "dataset_card.md").write_text(dataset_card, encoding="utf-8")

    print("SFT manifest + dataset card written.")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
