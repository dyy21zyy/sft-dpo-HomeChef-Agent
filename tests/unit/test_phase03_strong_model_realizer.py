"""TDD for Strong Model Chinese Surface Realizer (identity-preserving placeholders).

Covers:
  - 25-anchor catalog integrity
  - identity-preserving placeholderization (indexed, no swap/merge/loss)
  - business-fact preservation (e.g. "六点→晚上七点" stays 18:00→19:00)
  - Anchor Similarity Gate (entity-swap rejection, distinct pass)
  - real retry on anchor-similarity FAIL (max_attempts=3)
  - placeholder loss rejection + retry
"""
from __future__ import annotations

from homechef_booking.data.strong_model_realizer import (
    PLACEHOLDER_PATTERN,
    StrongModelSurfaceRealizer,
    _placeholderize_text,
    _restore_placeholders,
    anchor_similarity_gate,
    count_placeholders,
)
from homechef_booking.data.style_anchors import (
    STYLE_ANCHORS,
    anchor_normalized,
    anchor_skeleton,
    select_anchors,
)


class FakeBackend:
    """Deterministic mock backend that returns a given sequence of outputs."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, system: str, user: str) -> str:
        out = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return out


# ── 25-anchor catalog ────────────────────────────────────────

def test_anchor_catalog_integrity():
    assert len(STYLE_ANCHORS) == 25
    ids = [a.id for a in STYLE_ANCHORS]
    assert ids == list(range(1, 26))
    for a in STYLE_ANCHORS:
        assert a.difficulty in ("easy", "medium", "hard")
        assert a.text


def test_select_anchors_prefers_same_difficulty():
    anchors = select_anchors("hard", "requery_dietary", max_count=4)
    assert 2 <= len(anchors) <= 4
    assert any(a.difficulty == "hard" for a in anchors)
    assert len({a.id for a in anchors}) == len(anchors)


# ── Identity-preserving placeholderization ───────────────────

def test_placeholderize_two_distinct_times_get_distinct_indices():
    """Two different time values must get distinct indexed placeholders."""
    text = "六点有点早，改成晚上七点吧。"
    # Explicit (surface_value, type) pairs: both are TIME.
    slot_values = [("六点", "TIME"), ("晚上七点", "TIME")]
    placeholderized, mapping = _placeholderize_text(text, slot_values)
    assert "<TIME_1>" in placeholderized
    assert "<TIME_2>" in placeholderized
    # The two times are bound to distinct identity indices with distinct values.
    time1_value = mapping.get("<TIME_1>")
    time2_value = mapping.get("<TIME_2>")
    assert time1_value is not None and time2_value is not None
    assert time1_value != time2_value


def test_restore_preserves_distinct_values():
    """Restore maps each index back to its exact value — no swap/merge."""
    text = "六点有点早，改成晚上七点吧。"
    slot_values = [("六点", "TIME"), ("晚上七点", "TIME")]
    placeholderized, mapping = _placeholderize_text(text, slot_values)
    restored = _restore_placeholders(placeholderized, mapping)
    # Business fact preserved: the first time stays "六点", second stays "晚上七点"
    assert "六点有点早" in restored
    assert "改成晚上七点" in restored


def test_count_placeholders_indexed():
    """count_placeholders counts by TYPE regardless of index."""
    text = "把<TIME_1>改成<TIME_2>，<PEOPLE_1>人"
    counts = count_placeholders(text)
    assert counts["TIME"] == 2
    assert counts["PEOPLE"] == 1


# ── Business-fact preservation (18:00 → 19:00) ───────────────

def test_business_fact_time_change_preserved():
    """'六点有点早，改成晚上七点' must stay 18:00 → 19:00, never 19:00 → 19:00.

    The two time values are bound to distinct indices; restore must keep them
    distinct and each bound to its exact value (no merge/swap/loss).
    """
    raw_text = "六点有点早，改成晚上七点吧，其他都不变。"
    slot_values = [("六点", "TIME"), ("晚上七点", "TIME")]
    placeholderized, mapping = _placeholderize_text(raw_text, slot_values)
    # Both distinct times got indexed placeholders.
    time_values = sorted(mapping.values())
    assert time_values == sorted(["六点", "晚上七点"]), f"got {time_values}"
    # CRITICAL: the two bound time values must be distinct.
    assert len(set(mapping.values())) == 2
    assert mapping["<TIME_1>"] != mapping["<TIME_2>"]

    # Model's natural rewrite preserves the two indexed placeholders.
    rewritten = "<TIME_1>有点早，那就调到<TIME_2>吧。"
    restored = _restore_placeholders(rewritten, mapping)
    # Both original time surfaces appear exactly once, distinct.
    assert restored.count("六点") == 1
    assert restored.count("晚上七点") == 1


def test_business_fact_time_change_via_realizer():
    """End-to-end via realizer: two distinct times never collapse to one."""
    raw_text = "六点有点早，改成晚上七点吧，其他都不变。"
    slot_values = [("六点", "TIME"), ("晚上七点", "TIME")]
    backend = FakeBackend([
        "<TIME_1>有点早，那就调到<TIME_2>吧。",
    ])
    realizer = StrongModelSurfaceRealizer(backend, similarity_threshold=0.85, max_attempts=3)
    outcome = realizer.realize(
        raw_text=raw_text,
        slot_values=slot_values,
        difficulty="medium",
        scenario="state_inheritance",
        language_style="礼貌口语",
    )
    assert outcome.success is True, outcome.reason
    # First time (18:00 surface "六点") and second time (19:00 "晚上七点")
    assert "六点" in outcome.user_input
    assert "晚上七点" in outcome.user_input


# ── Anchor Similarity Gate ───────────────────────────────────

def test_anchor_similarity_gate_rejects_entity_swap():
    # Approved anchor 18: "不是明天，我刚才说错了，是后天晚上七点，其他条件别改。"
    # A pure entity-swap keeps the same words/structure → must be rejected.
    swap = "不是今天，我刚才说错了，是明天晚上八点，其他条件别改。"
    gate = anchor_similarity_gate(swap, "hard", "relative_correction", similarity_threshold=0.85)
    assert gate.passed is False, "entity-swap should be rejected"
    assert gate.best_anchor_id == 18


def test_anchor_similarity_gate_passes_distinct_paraphrase():
    # Genuinely different surface + structure → passes.
    distinct = "我记岔了，安排在下周三的中午吧，麻烦重新排。"
    gate = anchor_similarity_gate(distinct, "hard", "relative_correction", similarity_threshold=0.85)
    assert gate.passed is True


# ── Realizer placeholder preservation ────────────────────────

def test_realizer_placeholder_preservation_and_restore():
    raw_text = "把日期改到2026-08-15，7人，上海市静安区，川菜。"
    slot_values = {
        "service_date": "2026-08-15",
        "people": "7",
        "address": "上海市静安区",
        "cuisine": "川菜",
    }
    backend = FakeBackend([
        "日期换成<DATE_1>那天，用餐<PEOPLE_1>人，位置还是<ADDRESS_1>，口味<CUISINE_1>照旧。"
    ])
    realizer = StrongModelSurfaceRealizer(backend, similarity_threshold=0.85, max_attempts=3)
    outcome = realizer.realize(
        raw_text=raw_text,
        slot_values=slot_values,
        difficulty="medium",
        scenario="modify_time",
        language_style="普通口语",
    )
    assert outcome.success is True, outcome.reason
    assert "2026-08-15" in outcome.user_input
    assert "7" in outcome.user_input


def test_realizer_rejects_placeholder_loss():
    """Realizer retries if model drops a placeholder (identity set mismatch)."""
    raw_text = "改成6人，还是川菜。"
    slot_values = {"people": "6", "cuisine": "川菜"}
    backend = FakeBackend([
        "改成<PEOPLE_1>人吧。",  # drops <CUISINE_1>
        "改成<PEOPLE_1>人，还是<CUISINE_1>。",
    ])
    realizer = StrongModelSurfaceRealizer(backend, similarity_threshold=0.85, max_attempts=3)
    outcome = realizer.realize(
        raw_text=raw_text,
        slot_values=slot_values,
        difficulty="medium",
        scenario="modify_people",
        language_style="普通口语",
    )
    assert outcome.success is True
    assert outcome.attempts == 2


def test_realizer_rejects_placeholder_merge():
    """Realizer retries if model merges two identity-bound placeholders."""
    raw_text = "六点有点早，改成晚上七点吧。"
    slot_values = [("六点", "TIME"), ("晚上七点", "TIME")]
    backend = FakeBackend([
        "改成<TIME_1>吧。",  # merges <TIME_2> into <TIME_1> → identity set mismatch
        "<TIME_1>有点早，改成<TIME_2>吧。",
    ])
    realizer = StrongModelSurfaceRealizer(backend, similarity_threshold=0.85, max_attempts=3)
    outcome = realizer.realize(
        raw_text=raw_text,
        slot_values=slot_values,
        difficulty="medium",
        scenario="state_inheritance",
        language_style="礼貌口语",
    )
    assert outcome.success is True
    assert outcome.attempts == 2


# ── Anchor FAIL → real retry (must call backend again) ───────

def test_realizer_retries_on_anchor_copy():
    """Anchor FAIL must trigger a REAL retry (another backend call), not end."""
    raw_text = "把日期改成2026-08-14，6人，其他不变。"
    slot_values = {"service_date": "2026-08-14", "people": "6"}
    backend = FakeBackend([
        # Contains placeholders (passes identity validation) but is too similar
        # to approved anchor 18 "不是明天，我刚才说错了，是后天晚上七点，其他条件别改。"
        # → anchor similarity FAIL → real retry.
        "不是今天，我刚才说错了，是<DATE_1>晚上<PEOPLE_1>点，其他条件别改。",
        "我改主意了，就换成<DATE_1>，<PEOPLE_1>人吧，别的不动。",
    ])
    realizer = StrongModelSurfaceRealizer(backend, similarity_threshold=0.85, max_attempts=3)
    outcome = realizer.realize(
        raw_text=raw_text,
        slot_values=slot_values,
        difficulty="hard",
        scenario="relative_correction",
        language_style="口语化纠正",
    )
    assert outcome.success is True
    assert outcome.attempts == 2  # second attempt (real retry)
    assert backend.calls == 2  # backend was called TWICE
    assert "2026-08-14" in outcome.user_input


def test_realizer_fails_after_max_attempts():
    """All attempts are anchor copies → fail after max_attempts (3 backend calls)."""
    raw_text = "把日期改成2026-08-14，其他不变。"
    slot_values = {"service_date": "2026-08-14"}
    backend = FakeBackend([
        # always too similar to approved anchor 18
        "不是今天，我刚才说错了，是<DATE_1>晚上七点，其他条件别改。",
    ])
    realizer = StrongModelSurfaceRealizer(backend, similarity_threshold=0.85, max_attempts=3)
    outcome = realizer.realize(
        raw_text=raw_text,
        slot_values=slot_values,
        difficulty="hard",
        scenario="relative_correction",
        language_style="口语化纠正",
    )
    assert outcome.success is False
    assert outcome.attempts == 3
    assert backend.calls == 3  # real retries happened


# ── Helpers ─────────────────────────────────────────────────

def test_anchor_skeleton_normalized_helpers():
    assert anchor_skeleton("今天天气真好") == "X"
    assert anchor_skeleton("明天晚上6点，4个人") == "XNX，NX"
    assert anchor_normalized("今天，天气：真好。") == "今天天气真好"


def test_placeholder_pattern_matches_indexed_only():
    """Non-indexed <DATE> (no _N) must NOT match the identity pattern."""
    assert not PLACEHOLDER_PATTERN.search("<DATE>")
    assert PLACEHOLDER_PATTERN.search("<DATE_1>")
    assert PLACEHOLDER_PATTERN.search("<TIME_2>")


# ── Non-overlapping placeholderization (NO global str.replace) ──

def test_people_does_not_pollute_year():
    """people=2 must NOT be replaced inside the date "2026年8月15日"."""
    raw = "2026年8月15日，2个人。"
    slot_values = {"people": "2", "service_date": "2026年8月15日"}
    ph, mapping = _placeholderize_text(raw, slot_values)
    # The full date span is replaced by a single <DATE_1>; "2026" is NOT broken
    # up by a <PEOPLE_> token (no global str.replace of "2").
    assert "<DATE_1>" in ph
    assert ph.count("<PEOPLE_") == 1  # only the real "2个人" got a PEOPLE placeholder
    assert "<PEOPLE_1>" in ph
    assert mapping["<PEOPLE_1>"] == "2"
    # The placeholderized text must not contain a PEOPLE token inside the date.
    assert "<DATE_1>，<PEOPLE_1>个人。" == ph or "2026" not in ph


def test_people_does_not_pollute_time():
    """people=2 must NOT be replaced inside "12点"."""
    raw = "12点开饭，2个人。"
    slot_values = {"people": "2", "start_time": "12点"}
    ph, mapping = _placeholderize_text(raw, slot_values)
    # "12点" is replaced as a single <TIME_1>; the "2" inside "12" is NOT
    # hijacked by the people placeholder.
    assert "<TIME_1>开饭" in ph
    assert ph.count("<PEOPLE_") == 1  # only the real "2个人" → PEOPLE
    assert "<PEOPLE_1>" in ph
    assert mapping["<PEOPLE_1>"] == "2"
    assert "12点" not in ph or "<TIME_1>" in ph


def test_old_new_time_identity_preserved_nonoverlap():
    """Old (六点=18:00) and new (七点=19:00) stay distinct, non-overlapping."""
    raw = "六点有点早，改到七点吧。"
    slot_values = [("六点", "TIME"), ("七点", "TIME")]
    ph, mapping = _placeholderize_text(raw, slot_values)
    assert "<TIME_1>" in ph and "<TIME_2>" in ph
    assert mapping["<TIME_1>"] != mapping["<TIME_2>"]
    restored = _restore_placeholders(ph, mapping)
    assert "六点" in restored and "七点" in restored


def test_span_based_prevents_short_value_substring_match():
    """people=8 must NOT be broken out of the budget "800"."""
    raw = "预算800元，8个人。"
    slot_values = {"people": "8", "budget_max": "800"}
    ph, mapping = _placeholderize_text(raw, slot_values)
    # "800" is replaced as a single <BUDGET_1> (not broken by a PEOPLE token).
    assert "<BUDGET_1>元" in ph
    assert ph.count("<PEOPLE_") == 1  # only real "8个人" → PEOPLE
    assert mapping["<BUDGET_1>"] == "800"
    assert mapping["<PEOPLE_1>"] == "8"


# ── Approved anchors: 8/9/8 + 2-4 selection ─────────────────

def test_approved_anchors_distribution():
    from homechef_booking.data.style_anchors import DIFFICULTY_DISTRIBUTION
    assert DIFFICULTY_DISTRIBUTION["easy"] == 8
    assert DIFFICULTY_DISTRIBUTION["medium"] == 9
    assert DIFFICULTY_DISTRIBUTION["hard"] == 8
    assert sum(DIFFICULTY_DISTRIBUTION.values()) == 25


def test_anchor_count_always_2_to_4():
    from homechef_booking.data.style_anchors import STYLE_ANCHORS, select_anchors
    difficulties = ["easy", "medium", "hard"]
    scenarios = {
        s for s in {a.scenario for a in STYLE_ANCHORS}
    }
    for diff in difficulties:
        for scen in list(scenarios) + ["relative_time_correction", "unrelated_unknown"]:
            anchors = select_anchors(diff, scen, max_count=4)
            assert 2 <= len(anchors) <= 4, f"{diff}/{scen}: {len(anchors)}"
            assert len({a.id for a in anchors}) == len(anchors)  # no dup
    # Never returns empty.
    anchors = select_anchors("easy", "unrelated_unknown", max_count=4)
    assert len(anchors) >= 2


def test_anchors_prefer_same_difficulty():
    from homechef_booking.data.style_anchors import select_anchors
    anchors = select_anchors("hard", "unrelated_unknown", max_count=4)
    # At least 2 are hard (same difficulty), guaranteed by the 8 hard anchors.
    hard_ids = [a.id for a in anchors if a.difficulty == "hard"]
    assert len(hard_ids) >= 2


# ── Hard difficulty capability audit ─────────────────────────

def test_hard_anchors_require_multiple_capabilities():
    """Hard anchors must reflect >=2 combined business capabilities.

    Verify by scenario semantics: hard scenarios in the approved set combine
    modification + requery / correction + inheritance / multi-field / reversal.
    """
    from homechef_booking.data.style_anchors import STYLE_ANCHORS
    hard = [a for a in STYLE_ANCHORS if a.difficulty == "hard"]
    assert len(hard) == 8
    # Each hard scenario must be one of the recognized multi-capability patterns.
    multi_cap_scenarios = {
        "relative_time_correction",       # correction + date change + inheritance
        "modification_requires_requery",  # modify + re-query
        "tool_result_requery",            # tool result + date modify + re-query
        "tool_result_dietary_requery",    # tool result + dietary add + re-query
        "multi_field_modification",       # multiple field changes
        "candidate_rejection_research",   # rejection + re-search
        "confirmation_reversal",          # confirmation reversal + address change
        "multi_constraint_relative_time", # relative time + multiple constraints
    }
    for a in hard:
        assert a.scenario in multi_cap_scenarios, f"hard anchor {a.id} scenario '{a.scenario}' not multi-capability"
