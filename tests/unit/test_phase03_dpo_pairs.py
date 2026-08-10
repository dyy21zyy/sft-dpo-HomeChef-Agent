"""Task 6 Phase 03: DPO pair construction tests."""

from pathlib import Path

from homechef_booking.data.dpo_pairs import build_dpo_pairs
from homechef_booking.data.raw_validator import load_valid_raw_samples


def test_dpo_rejected_is_schema_valid_business_negative():
    raw = load_valid_raw_samples(Path("tests/fixtures/datasets/raw_smoke_candidate.jsonl"))[0]
    pairs = build_dpo_pairs(raw)
    assert len(pairs) >= 1
    for pair in pairs:
        assert pair.heuristic in {"H1", "H2", "H3", "H4", "H5", "H6", "H7"}
        assert pair.chosen != pair.rejected
        assert pair.raw_id == raw.id


def test_dpo_pair_has_chosen_rejected_sha256():
    raw = load_valid_raw_samples(Path("tests/fixtures/datasets/raw_smoke_candidate.jsonl"))[0]
    pairs = build_dpo_pairs(raw)
    for pair in pairs:
        assert len(pair.chosen_sha256) == 64
        assert len(pair.rejected_sha256) == 64


def test_dpo_prompt_uses_prompt_builder():
    raw = load_valid_raw_samples(Path("tests/fixtures/datasets/raw_smoke_candidate.jsonl"))[0]
    pairs = build_dpo_pairs(raw)
    for pair in pairs:
        assert len(pair.prompt) >= 2
        assert pair.prompt[0].role == "system"
