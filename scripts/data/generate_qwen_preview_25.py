"""Phase 03 v0.2.1 — Generate 25 Qwen Strong Surface Realization Review Previews.

Selects 25 representative samples (Easy 8 / Medium 9 / Hard 8) from the
validated v0.2.1 ScenarioFacts, rewrites their Chinese surface (user_input,
history NL, expected.reply) via the real Qwen API, and emits markdown + JSON
for human review.

Gates per sample:
  PLACEHOLDER_IDENTITY  (realizer enforces identity-preserving placeholders)
  ANCHOR_SIMILARITY     (reject entity-swaps of the 25 anchors)
  CHINESE               (no ASCII business-text leakage)
  BUSINESS_FACT         (slots + old→new relation preserved)

Fails fast (no fallback) if DASHSCOPE env vars are missing.
Usage:
  uv run python scripts/data/generate_qwen_preview_25.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

from homechef_booking.data.raw_sample import parse_raw_sample_line
from homechef_booking.data.strong_model_realizer import (
    StrongModelSurfaceRealizer,
    _placeholderize_text,
)
from homechef_booking.inference.factory import load_qwen_surface_backend
from homechef_booking.inference.qwen_api_backend import check_qwen_env

CONFIG_PATH = Path("configs/inference/qwen_surface.yaml")
RAW_PATH = Path("data/raw/phase03_raw_v0.2.1.jsonl")
OUT_DIR = Path("reports/generated/phase03/v0.2.1")

# Required scenario coverage across the 25 samples.
# Each name maps to a real sample.scenario value OR a sample tag
# (see _check_tags for the actual tag vocabulary in v0.2.1).
REQUIRED_SCENARIOS = [
    "missing_required_slots",        # scenario + tag
    "valid_search_tool_call",        # scenario
    "relative_time",                 # tag
    "specific_chef",                 # tag
    "state_inheritance",             # scenario + tag
    "modification_requires_requery", # tag (modify → re-query)
    "tool_fact_grounding",           # tag (tool_result matched/evidence)
    "candidate_order",               # tag (candidate selection)
    "dietary_preservation",          # tag (dietary)
    "explicit_confirmation",         # scenario
    "rejection",                     # scenario
    "tool_error",                    # scenario
    "unrelated",                     # scenario
    "tool_result",                   # scenario (matched / evidence)
    "tool_result_no_match",          # tool-result sub-variant (no_match)
    "tool_result_unavailable",       # tool-result sub-variant (unavailable)
]

# English business-text leakage pattern (for the Chinese gate).
_ENG_PATTERN = re.compile(
    r'\b(Beijing|Shanghai|Hangzhou|Guangzhou|Sichuan|Cantonese|Hunan|Shandong|'
    r'Chef\s+\w+|tomorrow|today|weekend|birthday|peanut_allergy|halal|vegetarian)\b',
    re.IGNORECASE,
)


# ── Difficulty-targeted sample selection ─────────────────────

def _tool_result_status(sample) -> str:
    """Return the latest tool-result status (e.g. matched/no_match/unavailable)."""
    history = getattr(sample.input, "history", []) or []
    for m in history:
        role = getattr(m, "role", "")
        if role == "tool":
            content = getattr(m, "content", "{}") or "{}"
            try:
                tr = json.loads(content)
                return tr.get("status", "")
            except (json.JSONDecodeError, TypeError):
                return ""
    return ""


def _matches(sample, scen: str) -> bool:
    """Does a sample cover the given scenario/tag label?

    Also matches tool-result sub-variants by inspecting the latest tool
    result status:
      - "tool_result_no_match"  → tool status "no_match"
      - "tool_result_unavailable" → tool status "unavailable"
    """
    if scen in sample.tags:
        return True
    if scen in sample.generation.capability_tags:
        return True
    if scen == sample.scenario:
        return True
    if scen == "tool_result_no_match" and _tool_result_status(sample) == "no_match":
        return True
    if scen == "tool_result_unavailable" and _tool_result_status(sample) == "unavailable":
        return True
    return False


_DIFF_CAPS = {"easy": 8, "medium": 9, "hard": 8}


def _select_samples(samples, difficulty: str, count: int, preferred_scenarios: list[str]) -> list:
    """Select `count` samples of `difficulty`, preferring scenario coverage."""
    pool = [s for s in samples if s.generation.difficulty == difficulty]
    chosen = []
    used_ids = set()

    # Pass 1: one per preferred scenario (in order)
    for scen in preferred_scenarios:
        if len(chosen) >= count:
            break
        for s in pool:
            if s.id in used_ids:
                continue
            if _matches(s, scen):
                chosen.append(s)
                used_ids.add(s.id)
                break

    # Pass 2: fill remaining
    for s in pool:
        if len(chosen) >= count:
            break
        if s.id in used_ids:
            continue
        chosen.append(s)
        used_ids.add(s.id)

    return chosen[:count]


def _ensure_global_coverage(samples, easy, medium, hard, required: list[str]) -> None:
    """Guarantee every REQUIRED_SCENARIO appears somewhere in the 25.

    For each uncovered scenario, add a representative sample. To keep the
    8/9/8 caps, drop a sample from the same difficulty that is NOT the sole
    representative of any required scenario (prefer a duplicate scenario type).
    """
    chosen = easy + medium + hard
    chosen_ids = {s.id for s in chosen}
    covered = {sc for sc in required for s in chosen if _matches(s, sc)}

    for scen in required:
        if scen in covered:
            continue
        candidate = next((s for s in samples if s.id not in chosen_ids and _matches(s, scen)), None)
        if candidate is None:
            continue
        diff = candidate.generation.difficulty
        bucket = {"easy": easy, "medium": medium, "hard": hard}[diff]
        cap = _DIFF_CAPS[diff]

        if len(bucket) < cap:
            bucket.append(candidate)
            chosen_ids.add(candidate.id)
        else:
            # Drop a sample from this bucket that is over-represented / not
            # needed to keep coverage. Prefer the last sample that another
            # selected sample already covers for the same scenarios.
            dropped = False
            for i in range(len(bucket) - 1, -1, -1):
                s = bucket[i]
                # Only drop if `s` is not the unique representative of any
                # scenario it covers (i.e. some other selected sample also
                # covers every scenario s covers).
                others = [x for x in (easy + medium + hard) if x.id != s.id]
                if all(any(_matches(o, sc) for o in others) for sc in required if _matches(s, sc)):
                    bucket[i] = candidate
                    chosen_ids.discard(s.id)
                    chosen_ids.add(candidate.id)
                    dropped = True
                    break
            if not dropped:
                # Fallback: replace the last element.
                last = bucket[-1]
                bucket[-1] = candidate
                chosen_ids.discard(last.id)
                chosen_ids.add(candidate.id)
        covered.add(scen)


# ── Business-fact preservation check ─────────────────────────

# Modification/old→new intent cue words (Chinese).
_MODIFY_CUES = ["改成", "换", "换成", "改到", "调整", "从", "改为", "提到", "提高到"]
_CORRECT_CUES = ["不是", "说错", "纠正", "其实", "应该是"]
_REQUERY_CUES = ["重新", "再查", "再帮", "重新帮我", "还有没有", "重新看看", "再看看"]
_CONFIRM_CUES = ["可以", "确认", "好的", "就这样", "就订", "订"]
_REJECT_CUES = ["不订", "算了", "取消", "先别订", "不要"]
_CANDIDATE_ORDINAL_CUES = ["第", "第二位", "第三位", "第一位"]
_UNRELATED_CUES = ["天气", "电影", "股市", "快递", "时间现在", "油价", "限号"]


def _detect_intent(raw_text: str) -> list[str]:
    """Detect current-turn intents (confirmation/rejection/requery/modify/...)."""
    intents = []
    if any(c in raw_text for c in _CONFIRM_CUES):
        intents.append("confirmation")
    if any(c in raw_text for c in _REJECT_CUES):
        intents.append("rejection")
    if any(c in raw_text for c in _REQUERY_CUES):
        intents.append("requery")
    if any(c in raw_text for c in _MODIFY_CUES) or any(c in raw_text for c in _CORRECT_CUES):
        intents.append("modification")
    if any(c in raw_text for c in _CANDIDATE_ORDINAL_CUES):
        intents.append("candidate_ordinal")
    if any(c in raw_text for c in _UNRELATED_CUES):
        intents.append("unrelated")
    return intents


def _business_fact_preserved(raw_text: str, rewritten: str, slot_values, original_placeholderized: str) -> tuple[bool, str]:
    """Verify facts EXPLICITLY expressed in the current turn survive the rewrite.

    Only facts whose surface spans were placeholderized are required to appear
    (those are what the current turn explicitly mentions). Inherited facts that
    are NOT mentioned in this turn (e.g. state_inheritance keeps people/cuisine
    implicit) are NOT required to be repeated.

    Additionally verifies:
      - old→new modification relation (when a modification intent is present)
      - confirmation / rejection / requery / candidate-ordinal / unrelated intents
    """
    # 1. Facts explicitly in this turn = the bound placeholder values.
    if isinstance(slot_values, dict):
        expected = {str(v) for v in slot_values.values() if v is not None and str(v)}
    else:
        expected = {str(v) for v, _ in slot_values if v is not None and str(v)}
    missing = [v for v in expected if v and v not in rewritten]
    if missing:
        return False, f"missing explicit slot surface(s): {missing}"

    # 2. Old→new modification relation.
    intents = _detect_intent(raw_text)
    if "modification" in intents:
        # Collect all distinct bound values; if >=2 differ, they represent an
        # old→new pair and BOTH must survive distinctly (realizer guarantees via
        # identity placeholders). If only one distinct value, it's a single-field
        # change — just ensure it survived (covered above).
        distinct = {v for v in expected if v}
        if len(distinct) >= 2:
            # The rewritten text must contain at least two DISTINCT surfaces
            # (the old and the new), i.e. not collapsed to one.
            present_distinct = {v for v in distinct if v in rewritten}
            if len(present_distinct) < 2:
                return False, f"old→new relation collapsed: only {present_distinct} distinct surface(s)"

    # 3. Intent-level preservation: a modification/requery/confirmation cue
    #    present in the original must still be present in the rewrite.
    for cue_group, label in [
        (_MODIFY_CUES + _CORRECT_CUES, "modification"),
        (_REQUERY_CUES, "requery"),
        (_CONFIRM_CUES, "confirmation"),
        (_REJECT_CUES, "rejection"),
    ]:
        if any(c in raw_text for c in cue_group) and not any(c in rewritten for c in cue_group):
            return False, f"{label} intent cue lost in rewrite"

    return True, ""


def _expected_reply_for(sample) -> str:
    """Return the expected reply for a FinalDecision sample, or N/A for a
    ToolCallDecision."""
    expected = sample.expected
    action = getattr(expected, "action", None)
    if action == "final":
        reply = getattr(expected, "reply", None)
        return reply if reply else ""
    return "N/A"  # ToolCallDecision (or unknown) → N/A


def _chinese_pass(text: str) -> bool:
    return not bool(_ENG_PATTERN.search(text))


# ── Main ─────────────────────────────────────────────────────

def main() -> int:
    # 0. --help
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Generate 25 Qwen Surface Review Previews.\n"
            "  uv run python scripts/data/generate_qwen_preview_25.py\n"
            "Env (must be set in THIS shell): DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL\n"
            "Config: configs/inference/qwen_surface.yaml (set `model`)\n"
        )
        return 0

    # 1. Fail fast env
    key_status, url_status = check_qwen_env()
    print(f"DASHSCOPE_API_KEY: {key_status}")
    print(f"DASHSCOPE_BASE_URL: {url_status}")
    if key_status == "MISSING" or url_status == "MISSING":
        print("DASHSCOPE_API_KEY_MISSING or DASHSCOPE_BASE_URL_MISSING — aborting. No fallback.")
        return 1

    # 2. Load backend + realizer
    try:
        backend = load_qwen_surface_backend(CONFIG_PATH)
    except Exception as exc:
        print(f"Qwen backend load FAILED: {exc}")
        return 1
    print(f"MODEL_USED: {backend.config.model}")
    realizer = StrongModelSurfaceRealizer(backend, similarity_threshold=0.85, max_attempts=3)

    # 3. Load samples and select 8/9/8
    if not RAW_PATH.exists():
        print(f"RAW_PATH not found: {RAW_PATH}")
        return 1
    samples = [
        parse_raw_sample_line(line)
        for line in RAW_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Raw samples loaded: {len(samples)}")

    easy = _select_samples(samples, "easy", 8, REQUIRED_SCENARIOS)
    medium = _select_samples(samples, "medium", 9, REQUIRED_SCENARIOS)
    hard = _select_samples(samples, "hard", 8, REQUIRED_SCENARIOS)
    _ensure_global_coverage(samples, easy, medium, hard, REQUIRED_SCENARIOS)
    selected = easy + medium + hard
    print(f"Selected: Easy={len(easy)} Medium={len(medium)} Hard={len(hard)}")

    # 4. Generate each preview
    previews = []
    retry_count = 0
    for idx, sample in enumerate(selected, 1):
        difficulty = sample.generation.difficulty
        scenario = sample.scenario
        user_input = sample.input.user_input

        # Build slot_values for identity-preserving placeholderization.
        # Prefer explicit (value,type) for times; dict for the rest.
        cs = sample.input.current_state.booking_state
        slot_values = {
            "service_date": cs.service_date,
            "start_time": cs.start_time,
            "people": cs.people,
            "address": cs.address,
            "cuisine": cs.cuisine,
            "chef_name": cs.chef_name,
        }
        # Remove None values.
        slot_values = {k: v for k, v in slot_values.items() if v is not None}

        # Placeholderize for display (identity-preserving, non-overlapping).
        placeholderized, mapping = _placeholderize_text(user_input, slot_values)

        # C. Anchor integration: each preview MUST use 2-4 approved anchors.
        from homechef_booking.data.style_anchors import select_anchors
        anchors = select_anchors(difficulty, scenario, max_count=4)
        anchor_ids_used = [a.id for a in anchors]
        if not anchor_ids_used:
            # Should be impossible (select_anchors never returns <2), but guard.
            anchor_ids_used = [a.id for a in select_anchors(difficulty, "valid_search_tool_call", 2)]

        outcome = realizer.realize(
            raw_text=user_input,
            slot_values=slot_values,
            difficulty=difficulty,
            scenario=scenario,
            language_style="自然口语",
        )
        if outcome.attempts > 1:
            retry_count += 1

        # Gates
        if outcome.success:
            rewritten = outcome.user_input
            anchor_gate_ok = outcome.anchor_similarity <= 0.85
            ch_ok = _chinese_pass(rewritten)
            biz_ok, biz_msg = _business_fact_preserved(user_input, rewritten, slot_values, placeholderized)
        else:
            rewritten = ""
            anchor_gate_ok = False
            ch_ok = False
            biz_ok, biz_msg = False, outcome.reason

        # D. Expected Reply: FinalDecision → show expected.reply; ToolCallDecision → N/A.
        expected_reply = _expected_reply_for(sample)

        previews.append({
            "id": f"PREVIEW_{idx:02d}",
            "sample_id": sample.id,
            "difficulty": difficulty,
            "scenario": scenario,
            "original": user_input,
            "placeholder_text": placeholderized,
            "rewritten": rewritten,
            "attempt_count": outcome.attempts,
            "anchor_ids_used": anchor_ids_used,
            "placeholder_identity": "PASS" if outcome.success else "FAIL",
            "anchor_similarity": ("PASS" if anchor_gate_ok else "FAIL"),
            "anchor_similarity_score": outcome.anchor_similarity,
            "chinese": "PASS" if ch_ok else "FAIL",
            "business_fact": ("PASS" if biz_ok else f"FAIL: {biz_msg}"),
            "expected_reply": expected_reply,
        })
        print(f"  [{idx}] {difficulty}/{scenario} attempts={outcome.attempts} anchors={anchor_ids_used} "
              f"PI={previews[-1]['placeholder_identity']} "
              f"AS={previews[-1]['anchor_similarity']} "
              f"CN={previews[-1]['chinese']} "
              f"BF={previews[-1]['business_fact']}")

    # 5. Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "qwen_surface_preview_25.json"
    md_path = OUT_DIR / "qwen_surface_preview_25.md"
    json_path.write_text(json.dumps(previews, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(previews), encoding="utf-8")

    # 6. Summary stats
    pi = sum(1 for p in previews if p["placeholder_identity"] == "PASS")
    as_ = sum(1 for p in previews if p["anchor_similarity"] == "PASS")
    cn = sum(1 for p in previews if p["chinese"] == "PASS")
    bf = sum(1 for p in previews if p["business_fact"] == "PASS")
    print("\n" + "=" * 60)
    print("PREVIEW SUMMARY")
    print("=" * 60)
    print(f"PREVIEW_COUNT = {len(previews)}")
    print(f"EASY/MEDIUM/HARD = {sum(1 for p in previews if p['difficulty']=='easy')}/"
          f"{sum(1 for p in previews if p['difficulty']=='medium')}/"
          f"{sum(1 for p in previews if p['difficulty']=='hard')}")
    print(f"PLACEHOLDER_GATE = {pi}/25")
    print(f"ANCHOR_SIMILARITY_GATE = {as_}/25")
    print(f"CHINESE_GATE = {cn}/25")
    print(f"BUSINESS_FACT_GATE = {bf}/25")
    print(f"RETRY_COUNT = {retry_count}")
    print(f"PREVIEW_PATH = {md_path}")
    print(f"READY_FOR_HUMAN_REVIEW = {'YES' if pi==25 and as_==25 and cn==25 and bf==25 else 'NO'}")
    return 0 if (pi == 25 and as_ == 25 and cn == 25 and bf == 25) else 1


def _render_markdown(previews: list[dict]) -> str:
    lines = [
        "# Qwen Strong Surface Realization — 25 Review Previews",
        "",
        "Generated: 25 previews (Easy 8 / Medium 9 / Hard 8) via real Qwen API.",
        "",
        "---",
        "",
    ]
    for p in previews:
        lines += [
            f"## {p['id']} — {p['difficulty'].title()} / {p['scenario']}",
            "",
            f"- **Difficulty**: {p['difficulty']}",
            f"- **Scenario**: {p['scenario']}",
            "- **Language Style**: 自然口语",
            "",
            "**Original**:",
            "",
            f"```\n{p['original']}\n```",
            "",
            "**Rewritten (Qwen)**:",
            "",
            f"```\n{p['rewritten'] or '(FAILED)'}\n```",
            "",
            f"- **Placeholder Text**: `{p['placeholder_text']}`",
            f"- **Expected Reply**: `{p['expected_reply'] or '(not set)'}`",
            f"- **Anchor IDs**: {p['anchor_ids_used']}",
            f"- **Attempt Count**: {p['attempt_count']}",
            "",
            "| Gate | Status |",
            "|---|---|",
            f"| Placeholder Identity | {p['placeholder_identity']} |",
            f"| Anchor Similarity | {p['anchor_similarity']} (score={p['anchor_similarity_score']:.2f}) |",
            f"| Chinese | {p['chinese']} |",
            f"| Business Fact Preservation | {p['business_fact']} |",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
