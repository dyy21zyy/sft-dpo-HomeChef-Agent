"""Phase 03 raw dataset generation plan and prompt builder.

Supports --mode smoke (25 rows) and --mode full (600 rows).
Full mode requires --approved-schema-file for ChatGPT approval gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from homechef_booking.data.raw_sample import parse_raw_sample_line
from homechef_booking.data.raw_validator import validate_raw_jsonl


class GenerationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mode: str = "smoke"
    count: int = 25
    seed: int = 3001
    dataset_version: str = "phase03_v0.1"
    output_kind_distribution: dict[str, int] = Field(default_factory=lambda: {"final": 15, "tool_call": 10})
    conversation_kind_distribution: dict[str, int] = Field(default_factory=lambda: {"single_turn": 20, "multi_turn": 5})
    scenarios: list[str] = Field(default_factory=lambda: [
        "missing_required_slots", "valid_search_tool_call", "matched_candidates",
        "specific_available", "specific_unavailable", "no_match", "out_of_service_area",
        "tool_error", "candidate_selection", "explicit_confirmation",
        "mutation_after_confirmation", "unrelated",
    ])


def build_generation_prompt(plan: GenerationPlan) -> str:
    return "\n".join([
        "You are a synthetic data generator for the HomeChef Booking Agent.",
        "",
        "CONTRACT: BOOKING_MACHINE_CONTRACT_v1 (homechef-booking-v1)",
        "",
        "Generate exactly the requested number of valid HomeChef Booking Decision JSONL rows.",
        "Each row must be a valid RawBookingSample as defined by the Phase 03 schema.",
        "",
        "RULES:",
        "- All outputs must conform to the Phase 00 HomeChef Booking Contract.",
        "- All available_tools must include the canonical find_chefs ToolSpec when output_kind is tool_call.",
        "- Do NOT copy or derive from Frozen Test or Diagnostic Dev cases.",
        "- Frozen Test outputs must never be used as training data.",
        "- Do NOT use RAG, memory, skills, or multi-agent patterns.",
        "- Every row must pass Phase 00 contract validation.",
        "- Use contract_id: homechef-booking-v1 and source: synthetic.",
        "- All IDs must be unique, format: phase03-raw-{NNNNNN}.",
        "",
        f"MODE: {plan.mode}",
        f"COUNT: {plan.count}",
        f"SEED: {plan.seed}",
        f"DATASET_VERSION: {plan.dataset_version}",
        f"SCENARIOS: {', '.join(plan.scenarios)}",
        "",
        "Output only valid JSONL (one JSON object per line). No markdown fences, no commentary.",
    ])


def parse_generated_raw_lines(text: str) -> list:
    samples = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        samples.append(parse_raw_sample_line(line))
    return samples


def generate_smoke_raw(output_path: Path, count: int = 25, seed: int = 3001) -> Path:
    """Generate smoke raw data programmatically (deterministic, no LLM call)."""
    from homechef_booking.schemas.tools import FindChefsInput
    import random

    rng = random.Random(seed)
    find_chefs_tool = {
        "type": "function",
        "function": {
            "name": "find_chefs",
            "description": "Find chefs",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["chef_name", "service_date", "start_time", "people", "address", "cuisine", "budget_min", "budget_max", "menu", "ingredient_purchase", "dietary_constraints", "occasion"],
                "properties": {
                    "chef_name": {"type": ["string", "null"]},
                    "service_date": {"type": ["string", "null"], "pattern": r"^\d{4}-\d{2}-\d{2}$", "format": "date"},
                    "start_time": {"type": ["string", "null"], "pattern": r"^([01]\d|2[0-3]):[0-5]\d$"},
                    "people": {"type": ["integer", "null"]},
                    "address": {"type": ["string", "null"]},
                    "cuisine": {"type": ["string", "null"]},
                    "budget_min": {"type": ["number", "null"]},
                    "budget_max": {"type": ["number", "null"]},
                    "menu": {"type": "array", "items": {"type": "string"}},
                    "ingredient_purchase": {"type": ["boolean", "null"]},
                    "dietary_constraints": {"type": "array", "items": {"type": "string"}},
                    "occasion": {"type": ["string", "null"]},
                },
            },
        },
    }

    scenarios = rng.choices(
        ["missing_required_slots", "valid_search_tool_call", "matched_candidates",
         "specific_available", "specific_unavailable", "no_match", "candidate_selection",
         "explicit_confirmation", "unrelated"],
        k=count,
    )

    cuisines = ["川菜", "粤菜", "湘菜", "鲁菜", "东北菜", "杭帮菜", "淮扬菜"]
    dietary_opts = [[], ["不吃花生"], ["不吃辣"], ["不吃海鲜"], ["不吃花生", "不吃辣"]]
    occasions = [None, "生日", "聚会", "商务宴请", None, None]
    addresses = ["上海市徐汇区", "上海市浦东新区", "北京市朝阳区", "杭州市西湖区", "广州市天河区"]
    menu_items = ["水煮鱼", "回锅肉", "麻婆豆腐", "白切鸡", "剁椒鱼头", "葱烧海参", "锅包肉", "西湖醋鱼", "东坡肉"]

    lines = []
    for i in range(1, count + 1):
        sid = f"phase03-raw-{i:06d}"
        scenario = scenarios[i - 1]
        output_kind = "tool_call" if scenario == "valid_search_tool_call" else "final"
        conv_kind = "multi_turn" if scenario in ("matched_candidates", "specific_available", "specific_unavailable") else "single_turn"
        is_tool = output_kind == "tool_call"
        bs = {
            "service_date": f"2026-08-{rng.randint(10, 25):02d}" if is_tool else None,
            "start_time": f"{rng.randint(11, 20):02d}:00" if is_tool else None,
            "people": rng.randint(2, 8) if is_tool else None,
            "address": rng.choice(addresses) if is_tool else None,
            "cuisine": rng.choice(cuisines) if is_tool else None,
            "budget_min": float(rng.randint(300, 800)) if is_tool and rng.random() > 0.3 else None,
            "budget_max": float(rng.randint(800, 2000)) if is_tool and rng.random() > 0.3 else None,
            "menu": rng.sample(menu_items, rng.randint(1, 3)) if is_tool else [],
            "chef_id": None, "chef_name": None,
            "ingredient_purchase": rng.choice([True, False, None]) if is_tool else None,
            "dietary_constraints": rng.choice(dietary_opts),
            "occasion": rng.choice(occasions),
            "confirmation": None,
        }
        state = {
            "booking_state": bs,
            "chef_query_status": "not_checked",
            "candidate_chefs": [],
            "awaiting_confirmation": False,
        }
        inp = {
            "history": [],
            "current_state": state,
            "user_input": f"smoke test input {i} for {scenario}",
            "current_time": "2026-08-10 10:00",
            "available_tools": [find_chefs_tool] if is_tool else [],
        }
        if is_tool:
            exp = {
                "action": "tool_call", "tool_name": "find_chefs",
                "arguments": {k: v for k, v in bs.items() if k in {"chef_name", "service_date", "start_time", "people", "address", "cuisine", "budget_min", "budget_max", "menu", "ingredient_purchase", "dietary_constraints", "occasion"}},
            }
        else:
            is_unrelated = scenario == "unrelated"
            exp = {
                "action": "final",
                "booking_state": bs,
                "chef_query_status": "not_checked",
                "candidate_chefs": [],
                "info_complete": False,
                "unrelated": is_unrelated,
                "missing_info": ["service_date", "start_time", "people", "address"],
                "reply_type": "handoff" if is_unrelated else "ask_multiple_required_fields",
                "reply": "请补充用餐日期、开始时间、人数和服务地址。",
            }
        lines.append(json.dumps({
            "id": sid, "dataset_version": "phase03_v0.1", "contract_id": "homechef-booking-v1",
            "source": "synthetic", "scenario": scenario, "output_kind": output_kind,
            "conversation_kind": conv_kind, "tags": [scenario],
            "input": inp, "expected": exp,
            "generation": {
                "generator": "deterministic_smoke", "model": "smoke_v0",
                "seed": seed, "prompt_sha256": "s" * 64,
                "generated_at": "2026-08-10T00:00:00Z",
            },
            "review": {"status": "machine_validated", "reviewer": None, "notes": []},
            "dpo_targets": rng.sample(["H1", "H2", "H3", "H4", "H5", "H6", "H7"], rng.randint(1, 2)) if rng.random() > 0.3 else [],
        }, ensure_ascii=False, sort_keys=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
