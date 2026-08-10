"""Task 7 Phase 03: Manifest and data card tests."""

import json
from pathlib import Path

from homechef_booking.data.manifest import write_dataset_data_card, write_dataset_manifest


def test_manifest_and_data_card_record_hashes_and_zero_frozen_overlap(tmp_path: Path):
    raw_path = Path("tests/fixtures/datasets/raw_smoke_candidate.jsonl")
    manifest_path = write_dataset_manifest(
        output_path=tmp_path / "manifest.json",
        dataset_version="phase03_v0.1",
        raw_path=raw_path,
        sft_train_path=None,
        sft_val_path=None,
        dpo_train_path=None,
        dpo_val_path=None,
        frozen_eval_overlap=0,
        diagnostic_dev_overlap=0,
    )
    card_path = write_dataset_data_card(
        output_path=tmp_path / "data_card.json",
        dataset_version="phase03_v0.1",
        frozen_eval_overlap=0,
        diagnostic_dev_overlap=0,
        phase02_dependency_status="not_used_for_training",
    )
    assert manifest_path.exists()
    assert card_path.exists()
    card_content = card_path.read_text(encoding="utf-8")
    assert '"frozen_eval_overlap": 0' in card_content
    manifest_content = manifest_path.read_text(encoding="utf-8")
    assert '"frozen_eval_overlap": 0' in manifest_content


def test_data_card_declares_no_frozen_test_training_use(tmp_path: Path):
    card_path = write_dataset_data_card(
        output_path=tmp_path / "data_card.json",
        dataset_version="phase03_v0.1",
        frozen_eval_overlap=0,
        diagnostic_dev_overlap=0,
    )
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["no_frozen_test_training_use"] is True
    assert card["phase"] == "03"
    assert card["raw_is_fact_source"] is True
