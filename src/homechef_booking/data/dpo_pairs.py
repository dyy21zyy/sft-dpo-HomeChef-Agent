"""Phase 03 DPO hard-negative pair construction from validated raw samples."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from homechef_booking.data.dataset_schema import DpoPair, SftMessage
from homechef_booking.data.raw_sample import RawBookingSample
from homechef_booking.data.sft_render import canonical_decision_json
from homechef_booking.prompts import PromptBuilder


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def perturb_for_heuristic(raw: RawBookingSample, heuristic: str):
    expected = raw.expected.model_copy(deep=True)
    booking = expected.booking_state if hasattr(expected, "booking_state") else None

    if heuristic == "H1" and booking:
        booking.chef_id = "chef_fake_999"
        booking.chef_name = "虚拟厨师"
    elif heuristic == "H2":
        if raw.output_kind == "final":
            from homechef_booking.schemas.decision import ToolCallDecision
            return ToolCallDecision(
                action="tool_call", tool_name="find_chefs",
                arguments={"chef_name": None, "service_date": None, "start_time": None,
                           "people": None, "address": None, "cuisine": None,
                           "budget_min": None, "budget_max": None, "menu": [],
                           "ingredient_purchase": None, "dietary_constraints": [], "occasion": None},
            )
    elif heuristic == "H3" and booking:
        booking.chef_id = "chef_stale_001"
        booking.service_date = "2026-01-01"
    elif heuristic == "H4" and booking:
        booking.dietary_constraints = []
    elif heuristic == "H5" and booking:
        booking.service_date = "2026-01-01"
        booking.start_time = None
    elif heuristic == "H6" and booking:
        booking.confirmation = True
        if hasattr(expected, "reply_type"):
            expected.reply_type = "booking_authorized"
    elif heuristic == "H7" and hasattr(expected, "candidate_chefs"):
        expected.candidate_chefs = list(reversed(expected.candidate_chefs)) if expected.candidate_chefs else []

    return expected


def build_dpo_pairs(raw: RawBookingSample) -> list[DpoPair]:
    pairs = []
    prompt_messages = PromptBuilder().build_messages(raw.input)
    sft_prompt = [SftMessage(role=str(msg["role"]), content=str(msg.get("content", ""))) for msg in prompt_messages]
    chosen = canonical_decision_json(raw.expected)
    for heuristic in raw.dpo_targets:
        if heuristic not in {"H1", "H2", "H3", "H4", "H5", "H6", "H7"}:
            continue
        rejected_obj = perturb_for_heuristic(raw, heuristic)
        rejected = canonical_decision_json(rejected_obj)
        if rejected == chosen:
            continue
        pairs.append(DpoPair(
            id=f"dpo-{raw.id}-{heuristic}",
            raw_id=raw.id,
            dataset_version=raw.dataset_version,
            heuristic=heuristic,
            prompt=sft_prompt,
            chosen=chosen,
            rejected=rejected,
            chosen_sha256=_sha256(chosen),
            rejected_sha256=_sha256(rejected),
            tags=list(raw.tags),
        ))
    return pairs


def build_dpo_dataset(raw_path: Path, train_path: Path, val_path: Path, val_ratio: float = 0.10, seed: int = 3001) -> list[DpoPair]:
    from homechef_booking.data.raw_validator import load_valid_raw_samples
    raw_samples = load_valid_raw_samples(raw_path)
    all_pairs = []
    for raw in raw_samples:
        all_pairs.extend(build_dpo_pairs(raw))
    import random
    rng = random.Random(seed)
    indices = list(range(len(all_pairs)))
    rng.shuffle(indices)
    split = max(1, int(len(all_pairs) * val_ratio))
    val_indices = set(indices[:split])
    train = [all_pairs[i] for i in indices if i not in val_indices]
    val = [all_pairs[i] for i in indices if i in val_indices]
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text("\n".join(json.dumps(p.model_dump(mode="json", exclude_none=False), ensure_ascii=False, sort_keys=True) for p in train), encoding="utf-8")
    val_path.write_text("\n".join(json.dumps(p.model_dump(mode="json", exclude_none=False), ensure_ascii=False, sort_keys=True) for p in val), encoding="utf-8")
    return all_pairs


# ── Targeted DPO ────────────────────────────────────────────────────────────

# Scenarios where H1 (fabricated chef identity) is meaningful:
# H1 mutates chef_id/chef_name — only applies when a chef is being selected/presented/confirmed
_H1_VALID_SCENARIOS = {
    "matched_candidates",      # presenting candidates → fabricating chef selection
    "candidate_selection",     # selecting a specific chef → fabricating selection
    "specific_available",      # chef available → fabricating identity
    "specific_unavailable",    # chef unavailable → fabricating identity
    "explicit_confirmation",   # confirming a booking → fabricating chef
}

# H1 must NOT be applied to pure tool_call samples (valid_search_tool_call)
# or scenarios without chef identity (missing_required_slots, unrelated, no_match, out_of_service_area, tool_error)


def build_targeted_dpo_pairs(raw: RawBookingSample, selected_targets: set[str]) -> list[DpoPair]:
    """Build DPO pairs only for selected high-risk heuristics, respecting H1 constraints."""
    pairs = []
    prompt_messages = PromptBuilder().build_messages(raw.input)
    sft_prompt = [SftMessage(role=str(msg["role"]), content=str(msg.get("content", ""))) for msg in prompt_messages]
    chosen = canonical_decision_json(raw.expected)
    for heuristic in raw.dpo_targets:
        if heuristic not in selected_targets:
            continue
        # Constraint 3: H1 only for final-decision chef-identity scenarios
        if heuristic == "H1":
            if raw.output_kind != "final":
                continue
            if raw.scenario not in _H1_VALID_SCENARIOS:
                continue
        rejected_obj = perturb_for_heuristic(raw, heuristic)
        rejected = canonical_decision_json(rejected_obj)
        if rejected == chosen:
            continue
        pairs.append(DpoPair(
            id=f"dpo-{raw.id}-{heuristic}",
            raw_id=raw.id,
            dataset_version=raw.dataset_version,
            heuristic=heuristic,
            prompt=sft_prompt,
            chosen=chosen,
            rejected=rejected,
            chosen_sha256=_sha256(chosen),
            rejected_sha256=_sha256(rejected),
            tags=list(raw.tags),
        ))
    return pairs


def build_dpo_targeted_dataset(
    raw_path: Path,
    train_path: Path,
    val_path: Path,
    selected_targets: list[str],
    min_per_target: int = 25,
    max_pairs: int = 240,
    target_total_min: int = 180,
    val_ratio: float = 0.10,
    seed: int = 3001,
) -> tuple[list[DpoPair], dict[str, int], dict[str, int]]:
    """Build a targeted DPO dataset sampling only high-risk heuristics.

    Returns:
        all_pairs: final selected DPO pairs
        target_available: raw counts of available pairs per target (before sampling)
        target_distribution: final distribution per target in the output
    """
    from homechef_booking.data.raw_validator import load_valid_raw_samples

    raw_samples = load_valid_raw_samples(raw_path)
    target_set = set(selected_targets)

    # Phase 1: build all eligible pairs grouped by target
    target_pools: dict[str, list[DpoPair]] = {t: [] for t in selected_targets}
    for raw in raw_samples:
        for pair in build_targeted_dpo_pairs(raw, target_set):
            target_pools[pair.heuristic].append(pair)

    target_available = {t: len(target_pools[t]) for t in selected_targets}

    # Phase 2: check feasibility
    total_available = sum(target_available.values())
    if total_available < target_total_min:
        raise ValueError(
            f"Total available targeted pairs ({total_available}) is below target_total_min ({target_total_min}). "
            f"Per-target available: {target_available}"
        )

    # Phase 3: deterministic sampling per target (min_per_target from each)
    import random
    rng = random.Random(seed)
    selected_pairs: list[DpoPair] = []
    for target in selected_targets:
        pool = list(target_pools[target])
        rng.shuffle(pool)
        take = min(min_per_target, len(pool))
        selected_pairs.extend(pool[:take])

    # Phase 4: if under max_pairs, fill remaining slots by round-robin from remaining pools
    remaining_by_target: dict[str, list[DpoPair]] = {}
    for target in selected_targets:
        taken_ids = {p.id for p in selected_pairs}
        remaining_by_target[target] = [p for p in target_pools[target] if p.id not in taken_ids]

    # Collect all remaining in shuffled order per target
    remaining_pool: list[DpoPair] = []
    for target in selected_targets:
        rem = list(remaining_by_target[target])
        rng.shuffle(rem)
        remaining_pool.extend(rem)

    rng.shuffle(remaining_pool)
    space_left = max_pairs - len(selected_pairs)
    if space_left > 0:
        selected_pairs.extend(remaining_pool[:space_left])

    # Phase 5: cap at max_pairs
    if len(selected_pairs) > max_pairs:
        rng.shuffle(selected_pairs)
        selected_pairs = selected_pairs[:max_pairs]

    # Phase 6: compute target distribution in final selection
    target_distribution: dict[str, int] = {}
    for pair in selected_pairs:
        target_distribution[pair.heuristic] = target_distribution.get(pair.heuristic, 0) + 1

    # Phase 7: train/val split
    indices = list(range(len(selected_pairs)))
    rng.shuffle(indices)
    split = max(1, int(len(selected_pairs) * val_ratio))
    val_indices = set(indices[:split])
    train = [selected_pairs[i] for i in indices if i not in val_indices]
    val = [selected_pairs[i] for i in indices if i in val_indices]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(
        "\n".join(json.dumps(p.model_dump(mode="json", exclude_none=False), ensure_ascii=False, sort_keys=True) for p in train),
        encoding="utf-8",
    )
    val_path.write_text(
        "\n".join(json.dumps(p.model_dump(mode="json", exclude_none=False), ensure_ascii=False, sort_keys=True) for p in val),
        encoding="utf-8",
    )

    return selected_pairs, target_available, target_distribution
