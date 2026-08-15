"""No-network unit tests for the Qwen smoke sample schema.

Confirms:
  - SMOKE_SAMPLES use the single field contract (difficulty/scenario/raw_text/
    slot_values/language_style/anchors).
  - No stale "original" / "placeholder_text" sample-field references remain.
  - The result-summary aggregation logic runs without KeyError.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from homechef_booking.data.strong_model_realizer import _placeholderize_text

# Load the smoke CLI module by file path (it is a script, not a package).
_SMOKE_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "data" / "smoke_qwen_surface_realizer.py"
_spec = importlib.util.spec_from_file_location("smoke_qwen_surface_realizer", _SMOKE_PATH)
_smoke_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_smoke_mod)

SMOKE_SAMPLES = _smoke_mod.SMOKE_SAMPLES

# Canonical sample contract keys (single source of truth).
CONTRACT_KEYS = {
    "difficulty",
    "scenario",
    "raw_text",
    "slot_values",
    "language_style",
    "anchors",
}


class _FakeOutcome:
    """Mimics RealizationOutcome for summary-logic testing (no network)."""

    def __init__(self, success: bool, user_input: str = "", attempts: int = 1):
        self.success = success
        self.user_input = user_input
        self.attempts = attempts
        self.max_similarity = 0.10  # passes gate
        self.reason = "ok"


def _run_summary_logic(samples):
    """Replicates the smoke script's per-sample + summary field access.

    This is the exact field-access pattern used by the smoke CLI after a
    successful realization. It must NOT KeyError.
    """
    results = []
    for sample in samples:
        # placeholderization + display (uses raw_text + slot_values).
        # Confirms raw_text + slot_values are readable (no KeyError) even though
        # the placeholderized string is not used in the result summary.
        _placeholderize_text(sample["raw_text"], sample["slot_values"])
        outcome = _FakeOutcome(True, user_input="改写结果。")
        restored = outcome.user_input

        results.append({
            "difficulty": sample["difficulty"],
            "scenario": sample["scenario"],
            "raw_text": sample["raw_text"],
            "rewritten": restored,
            "attempt_count": outcome.attempts,
            "anchor_sim": outcome.max_similarity,
        })
    return results


def test_smoke_sample_count():
    assert len(SMOKE_SAMPLES) == 3


def test_smoke_sample_contract_keys():
    """Every sample uses ONLY the canonical contract keys (no stale fields)."""
    for sample in SMOKE_SAMPLES:
        assert set(sample.keys()) == CONTRACT_KEYS, (
            f"sample keys mismatch: {set(sample.keys())} vs {CONTRACT_KEYS}"
        )


def test_smoke_sample_no_stale_fields():
    """No 'original' / 'placeholder_text' key anywhere in samples."""
    for sample in SMOKE_SAMPLES:
        assert "original" not in sample
        assert "placeholder_text" not in sample


def test_smoke_sample_required_values_present():
    for sample in SMOKE_SAMPLES:
        assert sample["difficulty"] in ("easy", "medium", "hard")
        assert isinstance(sample["scenario"], str) and sample["scenario"]
        assert isinstance(sample["raw_text"], str) and sample["raw_text"]
        assert sample["slot_values"] is not None
        assert isinstance(sample["anchors"], list) and sample["anchors"]


def test_summary_logic_no_keyerror():
    """The result-summary aggregation must run without KeyError."""
    results = _run_summary_logic(SMOKE_SAMPLES)
    assert len(results) == 3
    for r in results:
        assert "raw_text" in r
        assert "rewritten" in r
        assert "difficulty" in r
        assert "scenario" in r


def test_two_distinct_times_slot_values_are_list():
    """The medium sample with two times uses the list[(value,type)] contract."""
    medium = next(s for s in SMOKE_SAMPLES if s["difficulty"] == "medium")
    # It must be a list of (value, type) pairs for identity preservation.
    assert isinstance(medium["slot_values"], list)
    assert len(medium["slot_values"]) == 2
    types = {t for _, t in medium["slot_values"]}
    assert types == {"TIME"}
