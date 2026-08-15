"""Phase 03 v0.2.1 — Raw Generation Plan (specs → distribution).

Defines the scenario specifications and the formal 600-sample distribution,
which the Strong-Model direct Raw generator consumes. Difficulty and primary
scenario distribution are exact:

  missing / follow-up            = 130
  tool call                      = 140
  tool result → final            = 140
  confirmation / rejection / modification = 110
  unrelated / handoff            = 80
  ─────────────────────────────
  total                          = 600

  difficulty: Easy 180 / Medium 240 / Hard 180
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════
# Scenario Specification
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ScenarioSpec:
    """A single scenario specification for strong-model generation."""
    key: str                      # unique scenario key
    primary_category: str         # one of the 5 top-level categories
    difficulty: str               # easy | medium | hard
    description: str              # what the sample must exhibit
    required_slots: tuple[str, ...] = ()      # booking slots that must be present
    tool_mode: str = ""           # "" | "search" | "specific"
    tool_result_status: str = ""  # "" | matched/no_match/available/unavailable/not_found/out_of_service_area/error
    requires_tool_result: bool = False
    requires_requery: bool = False
    allowed_user_intents: tuple[str, ...] = ("modify", "search", "confirm", "reject", "unrelated", "followup")


# ═══════════════════════════════════════════════════════════
# 5 primary categories (formal distribution)
# ═══════════════════════════════════════════════════════════

# (key, difficulty, description)
_MISSING_FOLLOWUP_SPECS: list[tuple[str, str, str]] = [
    # Easy — single-slot missing
    ("missing_service_date", "easy", "User provides cuisine/people/address but NOT service_date; agent asks for date."),
    ("missing_start_time", "easy", "User provides date/people/address/cuisine but NOT start_time; agent asks for time."),
    ("missing_people", "easy", "User provides date/time/address but NOT people; agent asks for headcount."),
    ("missing_address", "easy", "User provides date/time/people/cuisine but NOT address; agent asks for address."),
    # Medium — partial state present, follow-up slot
    ("followup_fill_slot", "medium", "User fills a previously-missing slot while inheriting existing booking state."),
    # Hard — multiple missing slots / partial inheritance
    ("missing_multiple_slots", "hard", "User missing two or more required slots; agent asks for multiple fields."),
]

_TOOL_CALL_SPECS: list[tuple[str, str, str]] = [
    # Easy — complete search
    ("search_complete", "easy", "User provides all required slots; agent issues a search tool_call."),
    # Medium — relative time / specific chef / dietary
    ("search_relative_time", "medium", "User expresses a relative time (今天/明天/本周末/下周...); agent resolves deterministically and tool_calls."),
    ("search_specific_chef", "medium", "User requests a specific chef; agent tool_calls with chef_name."),
    ("search_dietary", "medium", "User includes dietary constraints; agent tool_calls preserving them."),
    # Hard — modification requires re-query / stale invalidation
    ("search_requery_after_modify", "hard", "User modifies a query dependency field (cuisine/date/people/address) after a prior query; agent re-issues tool_call and invalidates stale tool facts."),
]

_TOOL_RESULT_SPECS: list[tuple[str, str, str]] = [
    # Easy/Medium — matched / available
    ("tool_result_matched", "medium", "search matched; agent presents chef candidates."),
    ("tool_result_specific_available", "medium", "specific chef available; agent confirms."),
    # Medium — no_match / unavailable / not_found / out_of_service_area / error
    ("tool_result_no_match", "medium", "search no_match; agent informs no candidates."),
    ("tool_result_specific_unavailable", "medium", "specific chef unavailable; agent presents alternatives."),
    ("tool_result_not_found", "medium", "specific chef not found; agent informs."),
    ("tool_result_out_of_service_area", "medium", "address out of service area; agent informs."),
    ("tool_result_error", "hard", "tool error; agent pauses booking."),
]

_CONF_MOD_SPECS: list[tuple[str, str, str]] = [
    # Medium — single modification
    ("modify_date", "medium", "User modifies service_date; agent re-queries/inherits."),
    ("modify_time", "medium", "User modifies start_time."),
    ("modify_people", "medium", "User modifies people."),
    ("modify_address", "medium", "User modifies address."),
    ("modify_cuisine", "medium", "User modifies cuisine (query dependency)."),
    ("modify_budget", "medium", "User modifies budget."),
    # Medium — candidate selection
    ("candidate_selection", "medium", "User selects a candidate from tool result."),
    # Hard — multi-field modification / confirmation reversal / reject + research
    ("modify_multi_field", "hard", "User modifies two or more fields simultaneously."),
    ("confirmation_reversal", "hard", "User reverses a confirmation and modifies."),
    ("candidate_reject_research", "hard", "User rejects candidates and requests re-search."),
    # Easy/Medium — confirmation / rejection
    ("explicit_confirmation", "easy", "User affirms booking; agent authorizes."),
    ("rejection", "easy", "User declines; agent pauses."),
]

_UNRELATED_SPECS: list[tuple[str, str, str]] = [
    ("unrelated", "easy", "User asks a non-booking question; agent hands off."),
    ("handoff", "medium", "User intent is outside booking scope; agent hands off."),
]


@dataclass
class RawPlan:
    """Full 600-sample plan: list of ScenarioSpec with difficulty assignment."""
    specs: list[ScenarioSpec] = field(default_factory=list)


def _build_specs() -> list[ScenarioSpec]:
    specs: list[ScenarioSpec] = []

    def _add(specs_list, category):
        for (key, difficulty, desc) in specs_list:
            specs.append(ScenarioSpec(
                key=key,
                primary_category=category,
                difficulty=difficulty,
                description=desc,
            ))

    _add(_MISSING_FOLLOWUP_SPECS, "missing_followup")
    _add(_TOOL_CALL_SPECS, "tool_call")
    _add(_TOOL_RESULT_SPECS, "tool_result")
    _add(_CONF_MOD_SPECS, "confirmation_modification")
    _add(_UNRELATED_SPECS, "unrelated")
    return specs


ALL_SCENARIO_SPECS: list[ScenarioSpec] = _build_specs()

_SPEC_BY_KEY: dict[str, ScenarioSpec] = {s.key: s for s in ALL_SCENARIO_SPECS}

# Per-spec counts (sum = 600).
_SPEC_COUNTS: dict[str, int] = {
    # missing/followup (130)
    "missing_service_date": 30,
    "missing_start_time": 25,
    "missing_people": 20,
    "missing_address": 25,
    "followup_fill_slot": 15,
    "missing_multiple_slots": 15,
    # tool call (140)
    "search_complete": 40,
    "search_relative_time": 30,
    "search_specific_chef": 20,
    "search_dietary": 20,
    "search_requery_after_modify": 30,
    # tool result → final (140)
    "tool_result_matched": 30,
    "tool_result_specific_available": 20,
    "tool_result_no_match": 15,
    "tool_result_specific_unavailable": 20,
    "tool_result_not_found": 15,
    "tool_result_out_of_service_area": 15,
    "tool_result_error": 25,
    # confirmation / rejection / modification (110)
    "modify_date": 10,
    "modify_time": 10,
    "modify_people": 8,
    "modify_address": 8,
    "modify_cuisine": 10,
    "modify_budget": 5,
    "candidate_selection": 10,
    "modify_multi_field": 15,
    "confirmation_reversal": 10,
    "candidate_reject_research": 8,
    "explicit_confirmation": 8,
    "rejection": 8,
    # unrelated / handoff (80)
    "unrelated": 40,
    "handoff": 40,
}


def build_full_plan() -> RawPlan:
    """Build the ordered 600-sample plan (specs repeated per their counts)."""
    plan_specs: list[ScenarioSpec] = []
    for spec in ALL_SCENARIO_SPECS:
        count = _SPEC_COUNTS.get(spec.key, 0)
        plan_specs.extend([spec] * count)
    # Verify totals.
    assert len(plan_specs) == 600, f"plan size {len(plan_specs)} != 600"
    return RawPlan(specs=plan_specs)


def distribution_counts() -> dict[str, int]:
    """Primary-category distribution: {category: count}."""
    counts: dict[str, int] = {}
    for spec in ALL_SCENARIO_SPECS:
        counts[spec.primary_category] = counts.get(spec.primary_category, 0) + _SPEC_COUNTS.get(spec.key, 0)
    return counts


def difficulty_counts() -> dict[str, int]:
    """Difficulty distribution: {difficulty: count}."""
    counts: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    for spec in ALL_SCENARIO_SPECS:
        counts[spec.difficulty] = counts.get(spec.difficulty, 0) + _SPEC_COUNTS.get(spec.key, 0)
    return counts


def build_smoke_plan() -> RawPlan:
    """Build a 25-sample plan (Easy 8 / Medium 9 / Hard 8) covering core specs.

    This uses the SAME spec objects as the full 600 plan so the smoke shares
    the identical pipeline (prompt/backend/schema/validators).
    """
    easy_keys = [
        "missing_service_date", "missing_start_time", "missing_people", "missing_address",
        "search_complete", "explicit_confirmation", "rejection", "unrelated",
    ]
    medium_keys = [
        "followup_fill_slot", "search_relative_time", "search_specific_chef", "search_dietary",
        "tool_result_matched", "tool_result_no_match", "modify_time", "candidate_selection", "handoff",
    ]
    hard_keys = [
        "missing_multiple_slots", "search_requery_after_modify", "tool_result_error",
        "modify_multi_field", "confirmation_reversal", "candidate_reject_research",
        "tool_result_specific_unavailable", "tool_result_not_found",
    ]
    keys = easy_keys + medium_keys + hard_keys
    assert len(keys) == 25, f"smoke plan size {len(keys)} != 25"
    specs = [_SPEC_BY_KEY[k] for k in keys]
    return RawPlan(specs=specs)


__all__ = [
    "ScenarioSpec",
    "RawPlan",
    "ALL_SCENARIO_SPECS",
    "build_full_plan",
    "build_smoke_plan",
    "distribution_counts",
    "difficulty_counts",
]
