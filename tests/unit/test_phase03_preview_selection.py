"""No-network tests for the Qwen 25-preview sample-selection logic.

Confirms 8/9/8 difficulty split, 25 unique samples, and full required-scenario
coverage (including tool_result no_match / unavailable sub-variants) — without
any API call.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from homechef_booking.data.raw_sample import parse_raw_sample_line

_SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "data" / "generate_qwen_preview_25.py"
_spec = importlib.util.spec_from_file_location("gen_qwen_preview", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)

_RAW = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "phase03_raw_v0.2.1.jsonl"


def _load_samples():
    return [
        parse_raw_sample_line(line)
        for line in _RAW.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_required_scenarios_declared():
    assert len(_mod.REQUIRED_SCENARIOS) == 16


def test_selection_8_9_8_and_unique():
    samples = _load_samples()
    easy = _mod._select_samples(samples, "easy", 8, _mod.REQUIRED_SCENARIOS)
    medium = _mod._select_samples(samples, "medium", 9, _mod.REQUIRED_SCENARIOS)
    hard = _mod._select_samples(samples, "hard", 8, _mod.REQUIRED_SCENARIOS)
    _mod._ensure_global_coverage(samples, easy, medium, hard, _mod.REQUIRED_SCENARIOS)
    selected = easy + medium + hard
    assert len(easy) == 8
    assert len(medium) == 9
    assert len(hard) == 8
    assert len(selected) == 25
    assert len({s.id for s in selected}) == 25


def test_selection_full_scenario_coverage():
    samples = _load_samples()
    easy = _mod._select_samples(samples, "easy", 8, _mod.REQUIRED_SCENARIOS)
    medium = _mod._select_samples(samples, "medium", 9, _mod.REQUIRED_SCENARIOS)
    hard = _mod._select_samples(samples, "hard", 8, _mod.REQUIRED_SCENARIOS)
    _mod._ensure_global_coverage(samples, easy, medium, hard, _mod.REQUIRED_SCENARIOS)
    selected = easy + medium + hard
    covered = {
        sc
        for sc in _mod.REQUIRED_SCENARIOS
        for s in selected
        if _mod._matches(s, sc)
    }
    assert covered == set(_mod.REQUIRED_SCENARIOS), (
        f"missing: {set(_mod.REQUIRED_SCENARIOS) - covered}"
    )


def test_tool_result_subvariant_detection():
    """_matches detects no_match / unavailable tool-result sub-variants."""
    samples = _load_samples()
    no_match = [s for s in samples if _mod._matches(s, "tool_result_no_match")]
    unavailable = [s for s in samples if _mod._matches(s, "tool_result_unavailable")]
    assert no_match, "no_match sample not found"
    assert unavailable, "unavailable sample not found"


def test_markdown_render_no_error():
    """Markdown renderer runs for a minimal preview list (no API)."""
    previews = [
        {
            "id": "PREVIEW_01",
            "sample_id": "x",
            "difficulty": "easy",
            "scenario": "missing_required_slots",
            "original": "我想找个师傅。",
            "placeholder_text": "我想找个师傅。",
            "rewritten": "帮我安排一位师傅吧。",
            "attempt_count": 1,
            "anchor_ids_used": [1, 2],
            "placeholder_identity": "PASS",
            "anchor_similarity": "PASS",
            "anchor_similarity_score": 0.10,
            "chinese": "PASS",
            "business_fact": "PASS",
            "expected_reply": "",
        }
    ]
    md = _mod._render_markdown(previews)
    assert "PREVIEW_01" in md
    assert "帮我安排一位师傅吧。" in md
