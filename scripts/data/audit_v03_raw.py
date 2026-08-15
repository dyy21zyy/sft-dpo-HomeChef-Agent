"""Phase 03 v0.3 — Strict validation of the Strong-Model 600 Raw.

Runs schema, semantic, distribution, diversity, and contamination gates, then
writes reports under reports/generated/phase03/v0.3/.

The strong-model Raw is the single source of truth. No model is called.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

from homechef_booking.data.raw_sample import parse_raw_sample_line
from homechef_booking.data.raw_validator import validate_raw_jsonl

RAW_PATH = Path("data/raw/phase03_raw_strong_model_600_v0.3.jsonl")
FROZEN_PATH = Path("data/eval/frozen_test.jsonl")
DIAG_PATH = Path("data/dev/diagnostic_dev.jsonl")
OUT_DIR = Path("reports/generated/phase03/v0.3")

# Required distribution.
REQUIRED_GROUP = {"missing_followup": 130, "tool_call": 140, "tool_result_final": 140,
                  "confirm_reject_modify": 110, "unrelated_handoff": 80}
REQUIRED_DIFF = {"easy": 180, "medium": 240, "hard": 180}

_ENG_PATTERN = re.compile(
    r'\b(Beijing|Shanghai|Hangzhou|Guangzhou|Sichuan|Cantonese|Hunan|Shandong|'
    r'Chef\s+\w+|tomorrow|today|weekend|birthday|peanut_allergy|halal|vegetarian)\b',
    re.IGNORECASE,
)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Schema + Semantic ──
    report = validate_raw_jsonl(RAW_PATH, run_semantic=True)
    report_dict = {
        "total": report.total,
        "schema_valid": report.valid,
        "schema_errors": report.error_count,
        "semantic_valid": report.semantic_valid,
        "semantic_errors": report.semantic_error_count,
        "schema_error_samples": [
            {"row": e.row, "sample_id": e.sample_id, "path": e.path, "message": e.message}
            for e in report.errors
        ],
        "semantic_error_samples": [
            {"row": e.row, "sample_id": e.sample_id, "path": e.path, "message": e.message}
            for e in report.semantic_errors
        ],
        "semantic_applicability": report.semantic_applicability,
    }

    # ── 2. Distribution ──
    samples = [
        parse_raw_sample_line(line)
        for line in RAW_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    group_counter = Counter()
    diff_counter = Counter()
    scenario_counter = Counter()
    for s in samples:
        tags = set(s.tags)
        # Determine primary group from tags.
        group = None
        for g in ["missing_followup", "tool_call", "tool_result_final", "confirm_reject_modify", "unrelated_handoff"]:
            if g in tags:
                group = g
                break
        # Fallback to scenario mapping.
        if group is None:
            group = _group_from_scenario(s.scenario)
        group_counter[group] += 1
        # Difficulty from tags (difficulty_<level>) or generation.difficulty.
        d = None
        for lvl in ["easy", "medium", "hard"]:
            if f"difficulty_{lvl}" in tags:
                d = lvl
                break
        if d is None:
            d = s.generation.difficulty
        diff_counter[d] += 1
        scenario_counter[s.scenario] += 1

    distribution_report = {
        "group_distribution": dict(group_counter),
        "required_group": REQUIRED_GROUP,
        "group_match": {k: group_counter.get(k, 0) == v for k, v in REQUIRED_GROUP.items()},
        "difficulty_distribution": dict(diff_counter),
        "required_difficulty": REQUIRED_DIFF,
        "difficulty_match": {k: diff_counter.get(k, 0) == v for k, v in REQUIRED_DIFF.items()},
        "scenario_distribution": dict(scenario_counter.most_common()),
    }

    # ── 3. Diversity ──
    diversity_report = _diversity_audit(samples)

    # ── 4. Contamination ──
    contamination_report = _contamination_audit(samples)

    # ── 5. Write reports ──
    (OUT_DIR / "raw_validation_report.json").write_text(
        json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "distribution_report.json").write_text(
        json.dumps(distribution_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "diversity_report.json").write_text(
        json.dumps(diversity_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "contamination_report.json").write_text(
        json.dumps(contamination_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Summary ──
    print("RAW_VALIDATION_DONE")
    print(f"schema_valid={report.valid}/{report.total} schema_errors={report.error_count}")
    print(f"semantic_valid={report.semantic_valid}/{report.total} semantic_errors={report.semantic_error_count}")
    print(f"group_match={distribution_report['group_match']}")
    print(f"difficulty_match={distribution_report['difficulty_match']}")
    print(f"diversity_gate={diversity_report['gate']}")
    print(f"contamination_gate={contamination_report['gate']}")
    return 0


def _group_from_scenario(scenario: str) -> str:
    if scenario in ("unrelated", "handoff"):
        return "unrelated_handoff"
    if scenario in ("missing_required_slots", "missing_service_date", "missing_start_time",
                    "missing_people", "missing_address", "followup_slot_fill", "followup_fill_slot",
                    "state_inheritance", "state_inheritance_complete"):
        return "missing_followup"
    if scenario in ("valid_search_tool_call", "relative_time_search", "specific_chef_search",
                    "dietary_search", "search_complete", "search_relative_time", "search_specific_chef",
                    "search_dietary", "search_requery_after_modify"):
        return "tool_call"
    if scenario in ("tool_result_matched", "tool_result_no_match", "tool_result_specific_available",
                    "tool_result_specific_unavailable", "tool_result_not_found",
                    "tool_result_out_of_service_area", "tool_error", "specific_available",
                    "specific_unavailable", "specific_not_found", "out_of_service_area"):
        return "tool_result_final"
    if scenario in ("explicit_confirmation", "rejection", "confirmation_reversal",
                    "candidate_selection", "candidate_reject_research", "modify_date",
                    "modify_time", "modify_people", "modify_address", "modify_cuisine",
                    "modify_budget", "modify_multi_field", "modification_requires_requery",
                    "post_result_modification"):
        return "confirm_reject_modify"
    return "unknown"


def _diversity_audit(samples) -> dict:
    # Only count non-None user_input for NL uniqueness.
    all_ui = [s.input.user_input for s in samples if s.input.user_input]
    n = len(all_ui)
    exact_unique = len(set(all_ui))
    norm_set = set()
    for ui in all_ui:
        norm = re.sub(r'\d+', 'N', ui)
        norm = re.sub(r'[，。！？、；：\s]', '', norm)
        norm_set.add(norm)
    norm_unique = len(norm_set)

    def skeleton(t):
        sk = re.sub(r'[\u4e00-\u9fff]+', 'X', t)
        sk = re.sub(r'\d+', 'N', sk)
        return sk
    skels = Counter(skeleton(ui) for ui in all_ui)

    locations = set()
    cuisines = set()
    menus = set()
    dietary = set()
    chef_ids = set()
    rel_types = set()
    # Relative-time surface expressions (the strong-model Raw does NOT emit
    # relative_time_metadata; the expressions live in the user_input language).
    _RELATIVE_EXPR = ("今天", "今晚", "明天", "明晚", "后天", "这周", "下周", "本周末",
                      "这周六", "这周日", "下周一", "下周二", "下周三", "下周四", "下周五",
                      "下周六", "下周日", "这周五", "本周")
    for s in samples:
        bs = s.input.current_state.booking_state
        if bs.address:
            locations.add(bs.address)
        if bs.cuisine:
            cuisines.add(bs.cuisine)
        if bs.menu:
            menus.add(json.dumps(bs.menu, ensure_ascii=False))
        for d in bs.dietary_constraints:
            dietary.add(d)
        if bs.chef_id:
            chef_ids.add(bs.chef_id)
        rtm = s.generation.relative_time_metadata
        if rtm:
            rel_types.add(rtm.expression_type)
        # Also detect surface relative-time expressions in user_input.
        ui = s.input.user_input or ""
        for expr in _RELATIVE_EXPR:
            if expr in ui:
                rel_types.add(expr)

    dietary_singles = {d for d in dietary}
    # count combos (samples with >=2 dietary constraints)
    combos = sum(1 for s in samples if len(s.input.current_state.booking_state.dietary_constraints) >= 2)

    report = {
        "non_null_user_input": n,
        "exact_unique": exact_unique,
        "exact_unique_rate": round(exact_unique / n, 4) if n else 0,
        "normalized_unique": norm_unique,
        "normalized_unique_rate": round(norm_unique / n, 4) if n else 0,
        "skeleton_count": len(skels),
        "max_skeleton_frequency": max(skels.values()) if skels else 0,
        "locations": len(locations),
        "cuisines": len(cuisines),
        "menu_combinations": len(menus),
        "dietary_singles": len(dietary_singles),
        "dietary_combo_samples": combos,
        "chef_ids": len(chef_ids),
        "relative_time_types": len(rel_types),
        "relative_time_type_set": sorted(rel_types),
    }
    # Diversity gate: at least 300 exact unique, 240 normalized, 100 skeletons.
    report["gate"] = exact_unique >= 300 and norm_unique >= 240 and len(skels) >= 100
    return report


def _contamination_audit(samples) -> dict:
    from homechef_booking.data.contamination import check_raw_contamination
    frozen = check_raw_contamination(RAW_PATH, [FROZEN_PATH]) if FROZEN_PATH.exists() else None
    diag = check_raw_contamination(RAW_PATH, [DIAG_PATH]) if DIAG_PATH.exists() else None
    report = {
        "frozen_overlap": frozen.overlap_count if frozen else "file_missing",
        "frozen_overlapping_ids": frozen.overlapping_ids if frozen else [],
        "diagnostic_overlap": diag.overlap_count if diag else "file_missing",
        "diagnostic_overlapping_ids": diag.overlapping_ids if diag else [],
    }
    report["gate"] = (
        (frozen is not None and frozen.overlap_count == 0)
        and (diag is not None and diag.overlap_count == 0)
    )
    return report


if __name__ == "__main__":
    sys.exit(main())
