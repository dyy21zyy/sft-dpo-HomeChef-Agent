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
