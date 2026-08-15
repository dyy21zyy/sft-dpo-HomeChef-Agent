"""Phase 03 raw dataset generation plan and prompt builder.

Supports --mode smoke (25 rows) and --mode full (600 rows).
Full mode requires --approved-schema-file for ChatGPT approval gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from homechef_booking.data.raw_sample import parse_raw_sample_line


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


def _build_find_chefs_tool() -> dict:
    """Canonical find_chefs ToolSpec with 12 properties."""
    return {
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


def _empty_booking() -> dict:
    return {
        "service_date": None, "start_time": None, "people": None, "address": None,
        "cuisine": None, "budget_min": None, "budget_max": None, "menu": [],
        "chef_id": None, "chef_name": None, "ingredient_purchase": None,
        "dietary_constraints": [], "occasion": None, "confirmation": None,
    }


def _complete_booking(rng, addresses, cuisines, menu_items, dietary_opts, occasions) -> dict:
    return {
        "service_date": f"2026-08-{rng.randint(15, 28):02d}",
        "start_time": f"{rng.randint(11, 20):02d}:00",
        "people": rng.randint(2, 8),
        "address": rng.choice(addresses),
        "cuisine": rng.choice(cuisines),
        "budget_min": float(rng.randint(300, 800)),
        "budget_max": float(rng.randint(900, 2000)),
        "menu": rng.sample(menu_items, rng.randint(1, 3)),
        "chef_id": None, "chef_name": None,
        "ingredient_purchase": rng.choice([True, False, None]),
        "dietary_constraints": rng.choice(dietary_opts),
        "occasion": rng.choice(occasions),
        "confirmation": None,
    }


def _tool_call_arguments(bs: dict) -> dict:
    """Extract the 12 canonical find_chefs keys from a booking_state."""
    keys = ["chef_name", "service_date", "start_time", "people", "address",
            "cuisine", "budget_min", "budget_max", "menu", "ingredient_purchase",
            "dietary_constraints", "occasion"]
    return {k: bs[k] for k in keys}


def _make_history_with_tool_result(user_msg: str, args: dict, tool_result: dict, call_id: str = "call_s001") -> list:
    """Build a history sequence: user → assistant tool_call → tool result."""
    return [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function", "function": {
                "name": "find_chefs",
                "arguments": json.dumps(args, ensure_ascii=False),
            }},
        ]},
        {"role": "tool", "tool_call_id": call_id, "name": "find_chefs",
         "content": json.dumps(tool_result, ensure_ascii=False)},
    ]


_GENERATOR_META = {
    "generator": "deterministic_smoke",
    "model": "smoke_v0",
    "prompt_sha256": "0" * 64,
    "generated_at": "2026-08-10T00:00:00Z",
}


def generate_smoke_raw(output_path: Path, count: int = 25, seed: int = 3001) -> Path:
    """Generate smoke raw data programmatically (deterministic, no LLM call).

    Each scenario produces realistic Chinese HomeChef booking messages with
    expected behavior that genuinely matches the scenario label.
    """
    import random

    rng = random.Random(seed)
    find_chefs_tool = _build_find_chefs_tool()

    addresses = ["上海市徐汇区", "上海市浦东新区", "北京市朝阳区", "杭州市西湖区", "广州市天河区"]
    cuisines = ["川菜", "粤菜", "湘菜", "鲁菜", "东北菜", "杭帮菜", "淮扬菜"]
    dietary_opts = [[], ["不吃花生"], ["不吃辣"], ["不吃海鲜"], ["不吃花生", "不吃辣"]]
    occasions = [None, "生日", "聚会", "商务宴请", None, None]
    menu_items = ["水煮鱼", "回锅肉", "麻婆豆腐", "白切鸡", "剁椒鱼头", "葱烧海参", "锅包肉", "西湖醋鱼", "东坡肉"]

    # Fixed scenario distribution for 25 rows (deterministic, not random)
    scenario_plan = [
        "missing_required_slots",   # 1
        "missing_required_slots",   # 2
        "missing_required_slots",   # 3
        "valid_search_tool_call",   # 4
        "valid_search_tool_call",   # 5
        "valid_search_tool_call",   # 6
        "matched_candidates",       # 7
        "matched_candidates",       # 8
        "candidate_selection",      # 9
        "candidate_selection",      # 10
        "explicit_confirmation",    # 11
        "explicit_confirmation",    # 12
        "specific_available",       # 13
        "specific_available",       # 14
        "specific_unavailable",     # 15
        "specific_unavailable",     # 16
        "no_match",                 # 17
        "no_match",                 # 18
        "out_of_service_area",      # 19
        "tool_error",               # 20
        "unrelated",                # 21
        "unrelated",                # 22
        "unrelated",                # 23
        "missing_required_slots",   # 24
        "valid_search_tool_call",   # 25
    ]
    # Cycle the 25-scenario plan to reach the requested count
    while len(scenario_plan) < count:
        scenario_plan.extend(scenario_plan[: min(len(scenario_plan), count - len(scenario_plan))])

    lines = []
    for i in range(1, count + 1):
        sid = f"phase03-raw-{i:06d}"
        scenario = scenario_plan[i - 1]
        current_time = "2026-08-10 10:00"

        if scenario == "missing_required_slots":
            # Single-turn: user provides some info but genuinely missing required slots
            # Use i-based unique combo: 8 cuisines × 5 dietary × 4 occasions = 160 > ~96 rows
            user_inputs = [
                "想约一个家宴，麻烦帮我安排",
                "帮我预约私厨",
                "想吃川菜，帮忙找个厨师",
                "帮我找个厨师做一顿饭",
            ]
            user_input = user_inputs[(i - 1) % len(user_inputs)]
            bs = _empty_booking()
            _unique_occasions = [None, "生日", "聚会", "商务宴请"]
            bs["cuisine"] = cuisines[i % len(cuisines)]
            bs["dietary_constraints"] = dietary_opts[i % len(dietary_opts)]
            bs["occasion"] = _unique_occasions[i % len(_unique_occasions)]
            state = {"booking_state": bs, "chef_query_status": "not_checked",
                     "candidate_chefs": [], "awaiting_confirmation": False}
            inp = {"history": [], "current_state": state, "user_input": user_input,
                   "current_time": current_time, "available_tools": []}
            exp = {
                "action": "final", "booking_state": bs, "chef_query_status": "not_checked",
                "candidate_chefs": [], "info_complete": False, "unrelated": False,
                "missing_info": ["service_date", "start_time", "people", "address"],
                "reply_type": "ask_multiple_required_fields",
                "reply": "请补充用餐日期、开始时间、人数和服务地址。",
            }
            conv_kind = "single_turn"
            output_kind = "final"
            dpo_targets = ["H2", "H4"] if i % 3 == 0 else ["H2"]

        elif scenario == "valid_search_tool_call":
            # Single-turn: user provides complete required slots → tool_call
            user_inputs = [
                "8月15号晚上6点4个人，上海市徐汇区，川菜，不吃花生，生日宴",
                "8月20号中午12点6个人，北京市朝阳区，粤菜，商务宴请",
                "8月18号下午5点3个人，杭州市西湖区，杭帮菜，不吃海鲜",
                "8月22号下午1点5个人，广州市天河区，湘菜，聚会",
            ]
            user_input = user_inputs[(i - 1) % len(user_inputs)]
            bs = _complete_booking(rng, addresses, cuisines, menu_items, dietary_opts, occasions)
            state = {"booking_state": bs, "chef_query_status": "not_checked",
                     "candidate_chefs": [], "awaiting_confirmation": False}
            inp = {"history": [], "current_state": state, "user_input": user_input,
                   "current_time": current_time, "available_tools": [find_chefs_tool]}
            exp = {"action": "tool_call", "tool_name": "find_chefs",
                   "arguments": _tool_call_arguments(bs)}
            conv_kind = "single_turn"
            output_kind = "tool_call"
            dpo_targets = ["H1", "H5"] if i % 2 == 0 else ["H5"]

        elif scenario == "matched_candidates":
            # Multi-turn: after tool result matched, present candidates to user
            bs = _complete_booking(rng, addresses, cuisines, menu_items, dietary_opts, occasions)
            args = _tool_call_arguments(bs)
            candidates = [
                {"chef_id": "chef_001", "chef_name": "李师傅"},
                {"chef_id": "chef_002", "chef_name": "王师傅"},
            ]
            tool_result = {"mode": "search", "status": "matched", "candidates": candidates}
            history = _make_history_with_tool_result("帮我找个川菜厨师", args, tool_result)
            current_bs = {**bs, "chef_id": None, "chef_name": None}
            current_candidates = list(candidates)
            state = {"booking_state": current_bs, "chef_query_status": "matched",
                     "candidate_chefs": current_candidates, "awaiting_confirmation": False}
            inp = {"history": history, "current_state": state,
                   "user_input": "有哪些厨师可选？", "current_time": current_time,
                   "available_tools": []}
            exp = {
                "action": "final", "booking_state": current_bs, "chef_query_status": "matched",
                "candidate_chefs": current_candidates, "info_complete": True, "unrelated": False,
                "missing_info": [], "reply_type": "present_chef_candidates",
                "reply": "为您找到以下厨师：李师傅、王师傅，请选择一位。",
            }
            conv_kind = "multi_turn"
            output_kind = "final"
            dpo_targets = ["H7", "H1"]

        elif scenario == "candidate_selection":
            # Multi-turn: user selects a chef from candidates → confirm_specific_chef
            # chef_name must be consistent between current_state and expected to avoid
            # query dependency mutation (chef_name is in QUERY_DEPENDENCY_FIELDS)
            bs = _complete_booking(rng, addresses, cuisines, menu_items, dietary_opts, occasions)
            candidates = [
                {"chef_id": "chef_001", "chef_name": "李师傅"},
                {"chef_id": "chef_002", "chef_name": "王师傅"},
            ]
            selected = candidates[0]
            # current_state already has chef_name set (user specified in prior turn)
            current_bs = {**bs, "chef_id": None, "chef_name": selected["chef_name"]}
            state = {"booking_state": current_bs, "chef_query_status": "matched",
                     "candidate_chefs": candidates, "awaiting_confirmation": False}
            inp = {"history": [], "current_state": state,
                   "user_input": "就选李师傅吧", "current_time": current_time,
                   "available_tools": []}
            expected_bs = {**bs, "chef_id": selected["chef_id"], "chef_name": selected["chef_name"],
                           "confirmation": None}
            exp = {
                "action": "final", "booking_state": expected_bs, "chef_query_status": "matched",
                "candidate_chefs": candidates, "info_complete": True, "unrelated": False,
                "missing_info": [], "reply_type": "confirm_specific_chef",
                "reply": f"已选择{selected['chef_name']}，是否确认预约？",
            }
            conv_kind = "single_turn"
            output_kind = "final"
            dpo_targets = ["H1", "H3"]

        elif scenario == "explicit_confirmation":
            # Single-turn: awaiting_confirmation=true, user confirms → booking_authorized
            bs = _complete_booking(rng, addresses, cuisines, menu_items, dietary_opts, occasions)
            selected = {"chef_id": "chef_001", "chef_name": "李师傅"}
            current_bs = {**bs, "chef_id": selected["chef_id"], "chef_name": selected["chef_name"],
                          "confirmation": None}
            state = {"booking_state": current_bs, "chef_query_status": "matched",
                     "candidate_chefs": [selected], "awaiting_confirmation": True}
            inp = {"history": [], "current_state": state,
                   "user_input": "确认", "current_time": current_time,
                   "available_tools": []}
            expected_bs = {**current_bs, "confirmation": True}
            exp = {
                "action": "final", "booking_state": expected_bs, "chef_query_status": "matched",
                "candidate_chefs": [selected], "info_complete": True, "unrelated": False,
                "missing_info": [], "reply_type": "booking_authorized",
                "reply": "预约已授权，后续将有专员联系确认",
            }
            conv_kind = "single_turn"
            output_kind = "final"
            dpo_targets = ["H6"]

        elif scenario == "specific_available":
            # Multi-turn: user asks for specific chef by name, tool returns available
            bs = _complete_booking(rng, addresses, cuisines, menu_items, dietary_opts, occasions)
            bs["chef_name"] = "李师傅"
            args = _tool_call_arguments(bs)
            chef = {"chef_id": "chef_001", "chef_name": "李师傅"}
            tool_result = {"mode": "specific", "status": "available", "chef": chef}
            history = _make_history_with_tool_result("帮我查一下李师傅有没有空", args, tool_result)
            current_bs = {**bs, "chef_id": None, "chef_name": "李师傅"}
            state = {"booking_state": current_bs, "chef_query_status": "available",
                     "candidate_chefs": [chef], "awaiting_confirmation": False}
            inp = {"history": history, "current_state": state,
                   "user_input": "李师傅有空吗？", "current_time": current_time,
                   "available_tools": []}
            expected_bs = {**current_bs, "chef_id": chef["chef_id"]}
            exp = {
                "action": "final", "booking_state": expected_bs, "chef_query_status": "available",
                "candidate_chefs": [chef], "info_complete": True, "unrelated": False,
                "missing_info": [], "reply_type": "confirm_specific_chef",
                "reply": "李师傅在该时段有空，是否确认预约？",
            }
            conv_kind = "multi_turn"
            output_kind = "final"
            dpo_targets = ["H1", "H3"]

        elif scenario == "specific_unavailable":
            # Multi-turn: user asks for specific chef, tool returns unavailable + alternatives
            bs = _complete_booking(rng, addresses, cuisines, menu_items, dietary_opts, occasions)
            bs["chef_name"] = "李师傅"
            args = _tool_call_arguments(bs)
            alternatives = [
                {"chef_id": "chef_002", "chef_name": "王师傅"},
                {"chef_id": "chef_003", "chef_name": "张师傅"},
            ]
            tool_result = {"mode": "specific", "status": "unavailable",
                           "requested_chef": "李师傅", "alternatives": alternatives}
            history = _make_history_with_tool_result("帮我查一下李师傅有没有空", args, tool_result)
            current_bs = {**bs, "chef_id": None, "chef_name": "李师傅"}
            state = {"booking_state": current_bs, "chef_query_status": "unavailable",
                     "candidate_chefs": alternatives, "awaiting_confirmation": False}
            inp = {"history": history, "current_state": state,
                   "user_input": "李师傅没空的话有哪些替代？", "current_time": current_time,
                   "available_tools": []}
            exp = {
                "action": "final", "booking_state": current_bs, "chef_query_status": "unavailable",
                "candidate_chefs": alternatives, "info_complete": True, "unrelated": False,
                "missing_info": [], "reply_type": "present_alternatives",
                "reply": "李师傅在该时段无空，推荐以下替代厨师：王师傅、张师傅",
            }
            conv_kind = "multi_turn"
            output_kind = "final"
            dpo_targets = ["H7", "H4"]

        elif scenario == "no_match":
            # Multi-turn: tool result no_match → inform_no_match
            bs = _complete_booking(rng, addresses, cuisines, menu_items, dietary_opts, occasions)
            args = _tool_call_arguments(bs)
            tool_result = {"mode": "search", "status": "no_match", "candidates": []}
            history = _make_history_with_tool_result("帮我找个鲁菜厨师", args, tool_result)
            current_bs = {**bs, "chef_id": None, "chef_name": None}
            state = {"booking_state": current_bs, "chef_query_status": "no_match",
                     "candidate_chefs": [], "awaiting_confirmation": False}
            inp = {"history": history, "current_state": state,
                   "user_input": "没找到合适的厨师吗？", "current_time": current_time,
                   "available_tools": []}
            exp = {
                "action": "final", "booking_state": current_bs, "chef_query_status": "no_match",
                "candidate_chefs": [], "info_complete": True, "unrelated": False,
                "missing_info": [], "reply_type": "inform_no_match",
                "reply": "抱歉，未找到符合条件的厨师，请调整筛选条件再试。",
            }
            conv_kind = "multi_turn"
            output_kind = "final"
            dpo_targets = ["H4", "H5"]

        elif scenario == "out_of_service_area":
            # Multi-turn: tool result out_of_service_area → inform_out_of_service_area
            bs = _complete_booking(rng, addresses, cuisines, menu_items, dietary_opts, occasions)
            args = _tool_call_arguments(bs)
            tool_result = {"mode": "search", "status": "out_of_service_area", "candidates": []}
            history = _make_history_with_tool_result("帮我找个厨师", args, tool_result)
            current_bs = {**bs, "chef_id": None, "chef_name": None}
            state = {"booking_state": current_bs, "chef_query_status": "out_of_service_area",
                     "candidate_chefs": [], "awaiting_confirmation": False}
            inp = {"history": history, "current_state": state,
                   "user_input": "这个地址能服务吗？", "current_time": current_time,
                   "available_tools": []}
            exp = {
                "action": "final", "booking_state": current_bs,
                "chef_query_status": "out_of_service_area",
                "candidate_chefs": [], "info_complete": True, "unrelated": False,
                "missing_info": [], "reply_type": "inform_out_of_service_area",
                "reply": "抱歉，该地址不在服务范围内，请更换地址再试。",
            }
            conv_kind = "multi_turn"
            output_kind = "final"
            dpo_targets = ["H4"]

        elif scenario == "tool_error":
            # Multi-turn: tool result error → booking_paused (no retry per contract rules)
            bs = _complete_booking(rng, addresses, cuisines, menu_items, dietary_opts, occasions)
            args = _tool_call_arguments(bs)
            tool_result = {"mode": "search", "status": "error",
                           "error_code": "INTERNAL_ERROR", "retryable": True,
                           "message": "系统内部错误，请稍后重试"}
            history = _make_history_with_tool_result("帮我找个厨师", args, tool_result)
            current_bs = {**bs, "chef_id": None, "chef_name": None}
            state = {"booking_state": current_bs, "chef_query_status": "error",
                     "candidate_chefs": [], "awaiting_confirmation": False}
            inp = {"history": history, "current_state": state,
                   "user_input": "查询出错了？", "current_time": current_time,
                   "available_tools": []}
            exp = {
                "action": "final", "booking_state": current_bs, "chef_query_status": "error",
                "candidate_chefs": [], "info_complete": True, "unrelated": False,
                "missing_info": [], "reply_type": "booking_paused",
                "reply": "查询过程中出现错误，预约已暂停，请稍后重试。",
            }
            conv_kind = "multi_turn"
            output_kind = "final"
            dpo_targets = ["H5"]

        elif scenario == "unrelated":
            # Single-turn: user asks non-booking question → handoff, no booking slots asked
            # booking_state is complete (info_complete=true) so we don't ask for slots
            unrelated_inputs = [
                "今天天气怎么样",
                "给我讲个笑话",
                "帮我订一张机票",
            ]
            user_input = unrelated_inputs[(i - 1) % len(unrelated_inputs)]
            bs = _complete_booking(rng, addresses, cuisines, menu_items, dietary_opts, occasions)
            state = {"booking_state": bs, "chef_query_status": "not_checked",
                     "candidate_chefs": [], "awaiting_confirmation": False}
            inp = {"history": [], "current_state": state, "user_input": user_input,
                   "current_time": current_time, "available_tools": []}
            exp = {
                "action": "final", "booking_state": bs, "chef_query_status": "not_checked",
                "candidate_chefs": [], "info_complete": True, "unrelated": True,
                "missing_info": [],
                "reply_type": "handoff",
                "reply": "这不是上门私厨预约相关请求，已切换到对应服务",
            }
            conv_kind = "single_turn"
            output_kind = "final"
            dpo_targets = []

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        lines.append(json.dumps({
            "id": sid, "dataset_version": "phase03_v0.1", "contract_id": "homechef-booking-v1",
            "source": "synthetic", "scenario": scenario, "output_kind": output_kind,
            "conversation_kind": conv_kind, "tags": [scenario],
            "input": inp, "expected": exp,
            "generation": {
                "generator": _GENERATOR_META["generator"], "model": _GENERATOR_META["model"],
                "seed": seed, "prompt_sha256": _GENERATOR_META["prompt_sha256"],
                "generated_at": _GENERATOR_META["generated_at"],
            },
            "review": {"status": "machine_validated", "reviewer": None, "notes": []},
            "dpo_targets": dpo_targets,
        }, ensure_ascii=False, sort_keys=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ── v0.2 Full Generator (600 rows, reuses Scenario Lineage Builder) ────────


def _build_v02_sample(
    i: int,
    scenario_key: str,
    rng,
    seed: int,
    current_time: str = "2026-08-12 18:00",
) -> dict:
    """Build a single v0.2 raw sample dict from ScenarioFacts lineage.

    This is the shared builder used by both smoke (25) and full (600)
    generators.  Every sample is derived from a ScenarioFacts object.
    """
    from homechef_booking.data.raw_sample import (
        RelativeTimeMetadata,
        StateTransitionMetadata,
        ToolFactMetadata,
    )
    from homechef_booking.data.scenario_lineage import (
        ScenarioFacts,
        build_current_state,
        build_expected_final,
        build_expected_tool_call,
        build_history_with_tool_result,
    )
    from homechef_booking.data.tool_spec_factory import canonical_find_chefs_tool_dict
    from homechef_booking.schemas.booking import BookingSlot

    find_chefs_tool = canonical_find_chefs_tool_dict()
    sid = f"phase03-raw-v02-{i:06d}"
    seed_i = seed + i

    if scenario_key == "missing_required_slots":
        which = (i - 1) % 3
        bs_dict = _v02_empty_booking()
        user_inputs = [
            "想约一个家宴，麻烦帮我安排",
            "明天晚上需要厨师",
            "需要川菜厨师",
        ]
        user_input = user_inputs[which]
        if which == 1:
            bs_dict["service_date"] = "2026-08-13"
        elif which == 2:
            bs_dict["cuisine"] = "Sichuan"
        bs = BookingSlot.model_validate(bs_dict)
        facts = ScenarioFacts(booking_state=bs, scenario="missing_required_slots")
        history = []
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "single_turn"
        available_tools = []
        dpo_targets = ["H2"]
        capability_tags = ["missing_required_slots", "reply_policy"]

    elif scenario_key == "valid_search_tool_call":
        bs_dict = _v02_complete_booking(rng)
        relative_map = {
            "today": ("2026-08-12", "today"),
            "tomorrow": ("2026-08-13", "tomorrow"),
            "this_saturday": ("2026-08-15", "this_saturday"),
            "this_sunday": ("2026-08-16", "this_sunday"),
        }
        rel_types = list(relative_map.keys())
        rel_idx = (i - 1) % len(rel_types)
        rel_key = rel_types[rel_idx]
        bs_dict["service_date"] = relative_map[rel_key][0]
        rel_type = rel_key
        bs = BookingSlot.model_validate(bs_dict)
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="search",
            scenario="valid_search_tool_call",
        )
        history = []
        current_state = build_current_state(facts)
        expected = build_expected_tool_call(facts)
        output_kind = "tool_call"
        conv_kind = "single_turn"
        available_tools = [find_chefs_tool]
        dpo_targets = ["H5"]
        capability_tags = ["search_tool_call", "relative_time"]
        user_input = f"需要{bs.cuisine}厨师，{bs.service_date}，{bs.people}人，{bs.address}"

    elif scenario_key == "state_inheritance":
        bs_dict = _v02_complete_booking(rng)
        bs_dict["dietary_constraints"] = ["peanut_allergy"]
        mutation_type = (i - 1) % 3  # 0=date, 1=cuisine, 2=re-query
        if mutation_type == 1:
            bs_dict["cuisine"] = "Sichuan"
            user_input = "改成粤菜"
        elif mutation_type == 2:
            bs_dict["cuisine"] = "Sichuan"
            user_input = "再查一次"
        else:
            bs_dict["service_date"] = "2026-08-13"
            user_input = "改成8月14号"
        bs = BookingSlot.model_validate(bs_dict)
        expected_bs_dict = dict(bs_dict)
        if mutation_type == 1:
            expected_bs_dict["cuisine"] = "Cantonese"
        elif mutation_type == 0:
            expected_bs_dict["service_date"] = "2026-08-14"
        expected_bs = BookingSlot.model_validate(expected_bs_dict)
        facts = ScenarioFacts(
            booking_state=expected_bs,
            tool_mode="search",
            scenario="state_inheritance",
        )
        history = []
        current_state = {
            "booking_state": bs_dict,
            "chef_query_status": "not_checked",
            "candidate_chefs": [],
            "awaiting_confirmation": False,
        }
        expected = build_expected_tool_call(facts)
        output_kind = "tool_call"
        conv_kind = "multi_turn"
        available_tools = [find_chefs_tool]
        dpo_targets = ["H3", "H4"]
        capability_tags = ["state_inheritance", "modification_requires_requery"]

    elif scenario_key == "tool_result_matched":
        bs_dict = _v02_complete_booking(rng)
        bs = BookingSlot.model_validate(bs_dict)
        n_cands = 2 + (i % 3)  # 2-4 candidates
        chef_names = ["Chef Wang", "Chef Li", "Chef Zhao", "Chef Chen", "Chef Liu"]
        candidates = [
            {"chef_id": f"C{i:03d}", "chef_name": chef_names[j % len(chef_names)]}
            for j, i in enumerate(range(3, 3 + n_cands), 3)
        ]
        tool_result = {"mode": "search", "status": "matched", "candidates": candidates}
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="search",
            tool_result_status="matched",
            tool_result_payload=tool_result,
            scenario="tool_result",
        )
        user_input = f"找到{len(candidates)}位厨师"
        history = build_history_with_tool_result("find chefs", facts)
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = ["H1", "H7", "H8"]
        capability_tags = ["tool_result", "tool_fact_grounding", "reply_policy", "candidate_order"]

    elif scenario_key == "tool_result_specific_available":
        bs_dict = _v02_complete_booking(rng)
        chef_name = rng.choice(["Chef Zhang", "Chef Wang", "Chef Li"])
        bs_dict["chef_name"] = chef_name
        bs = BookingSlot.model_validate(bs_dict)
        chef_id = f"C{rng.randint(100, 999)}"
        chef = {"chef_id": chef_id, "chef_name": chef_name}
        tool_result = {"mode": "specific", "status": "available", "chef": chef}
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="specific",
            tool_result_status="available",
            tool_result_payload=tool_result,
            requested_chef_name=chef_name,
            scenario="tool_result",
        )
        user_input = f"{chef_name}有档期"
        history = build_history_with_tool_result(f"查一下{chef_name}有没有空", facts)
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = ["H1", "H7", "H8"]
        capability_tags = ["tool_result", "tool_fact_grounding", "reply_policy", "specific_chef"]

    elif scenario_key == "tool_result_specific_unavailable":
        bs_dict = _v02_complete_booking(rng)
        chef_name = rng.choice(["Chef Zhang", "Chef Wang", "Chef Li"])
        bs_dict["chef_name"] = chef_name
        bs = BookingSlot.model_validate(bs_dict)
        alt_names = ["Chef Chen", "Chef Liu", "Chef Zhao"]
        n_alts = 1 + (i % 2)
        alternatives = [
            {"chef_id": f"C{rng.randint(200, 899)}", "chef_name": alt_names[j % len(alt_names)]}
            for j in range(n_alts)
        ]
        tool_result = {
            "mode": "specific", "status": "unavailable",
            "requested_chef": chef_name, "alternatives": alternatives,
        }
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="specific",
            tool_result_status="unavailable",
            tool_result_payload=tool_result,
            requested_chef_name=chef_name,
            scenario="tool_result",
        )
        user_input = f"{chef_name}无档期，有哪些替代"
        history = build_history_with_tool_result(f"查一下{chef_name}有没有空", facts)
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = ["H7", "H8"]
        capability_tags = ["tool_result", "tool_fact_grounding", "reply_policy", "specific_chef"]

    elif scenario_key == "tool_result_no_match":
        bs_dict = _v02_complete_booking(rng)
        bs = BookingSlot.model_validate(bs_dict)
        tool_result = {"mode": "search", "status": "no_match", "candidates": []}
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="search",
            tool_result_status="no_match",
            tool_result_payload=tool_result,
            scenario="tool_result",
        )
        user_input = "没找到合适的厨师吗"
        history = build_history_with_tool_result("find chefs", facts)
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = ["H4"]
        capability_tags = ["tool_result", "reply_policy"]

    elif scenario_key == "tool_result_out_of_service_area":
        bs_dict = _v02_complete_booking(rng)
        bs = BookingSlot.model_validate(bs_dict)
        tool_result = {"mode": "search", "status": "out_of_service_area", "candidates": []}
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="search",
            tool_result_status="out_of_service_area",
            tool_result_payload=tool_result,
            scenario="tool_result",
        )
        user_input = "这个地址能服务吗"
        history = build_history_with_tool_result("find chefs", facts)
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = []
        capability_tags = ["tool_result", "reply_policy"]

    elif scenario_key == "dietary_modification":
        bs_dict = _v02_complete_booking(rng)
        bs_dict["dietary_constraints"] = ["peanut_allergy"]
        bs = BookingSlot.model_validate(bs_dict)
        expected_bs_dict = dict(bs_dict)
        expected_bs_dict["dietary_constraints"] = ["peanut_allergy", "halal"]
        expected_bs = BookingSlot.model_validate(expected_bs_dict)
        facts = ScenarioFacts(
            booking_state=expected_bs,
            tool_mode="search",
            scenario="valid_search_tool_call",
        )
        user_input = "找粤菜厨师，花生过敏，清真"
        history = []
        current_state = {
            "booking_state": bs_dict,
            "chef_query_status": "not_checked",
            "candidate_chefs": [],
            "awaiting_confirmation": False,
        }
        expected = build_expected_tool_call(facts)
        output_kind = "tool_call"
        conv_kind = "multi_turn"
        available_tools = [find_chefs_tool]
        dpo_targets = ["H4"]
        capability_tags = ["dietary_preservation", "modification_requires_requery", "state_inheritance"]

    elif scenario_key == "explicit_confirmation":
        bs_dict = _v02_complete_booking(rng)
        chef_name = rng.choice(["Chef Zhang", "Chef Wang"])
        chef_id = f"C{rng.randint(100, 999)}"
        bs_dict["chef_id"] = chef_id
        bs_dict["chef_name"] = chef_name
        bs = BookingSlot.model_validate(bs_dict)
        chef = {"chef_id": chef_id, "chef_name": chef_name}
        tool_result = {"mode": "specific", "status": "available", "chef": chef}
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="specific",
            tool_result_status="available",
            tool_result_payload=tool_result,
            requested_chef_name=chef_name,
            user_confirms=True,
            scenario="explicit_confirmation",
        )
        user_input = "确认"
        history = []
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = ["H6"]
        capability_tags = ["confirmation", "reply_policy"]

    elif scenario_key == "rejection":
        bs_dict = _v02_complete_booking(rng)
        chef_name = rng.choice(["Chef Zhang", "Chef Wang"])
        chef_id = f"C{rng.randint(100, 999)}"
        bs_dict["chef_id"] = chef_id
        bs_dict["chef_name"] = chef_name
        bs = BookingSlot.model_validate(bs_dict)
        chef = {"chef_id": chef_id, "chef_name": chef_name}
        tool_result = {"mode": "specific", "status": "available", "chef": chef}
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="specific",
            tool_result_status="available",
            tool_result_payload=tool_result,
            requested_chef_name=chef_name,
            user_rejects=True,
            scenario="rejection",
        )
        user_input = "算了"
        history = []
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = ["H6"]
        capability_tags = ["confirmation", "reply_policy"]

    elif scenario_key == "tool_error":
        bs_dict = _v02_complete_booking(rng)
        bs = BookingSlot.model_validate(bs_dict)
        tool_result = {
            "mode": "search", "status": "error",
            "error_code": "SERVICE_ERROR", "retryable": False,
            "message": "查询服务暂时不可用",
        }
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="search",
            tool_result_status="error",
            tool_result_payload=tool_result,
            scenario="tool_error",
        )
        user_input = "查询出错了"
        history = build_history_with_tool_result("find chefs", facts)
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = []
        capability_tags = ["tool_result", "reply_policy"]

    elif scenario_key == "unrelated":
        bs_dict = _v02_empty_booking()
        bs = BookingSlot.model_validate(bs_dict)
        facts = ScenarioFacts(booking_state=bs, scenario="unrelated")
        user_inputs = ["今天天气怎么样", "帮我推荐一部电影", "明天股市行情如何"]
        user_input = user_inputs[i % len(user_inputs)]
        history = []
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "single_turn"
        available_tools = []
        dpo_targets = []
        capability_tags = ["reply_policy"]

    elif scenario_key == "tool_result_specific_not_found":
        bs_dict = _v02_complete_booking(rng)
        chef_name = f"Chef Ghost{i}"
        bs_dict["chef_name"] = chef_name
        bs = BookingSlot.model_validate(bs_dict)
        tool_result = {
            "mode": "specific", "status": "not_found",
            "requested_chef": chef_name, "alternatives": [],
        }
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="specific",
            tool_result_status="not_found",
            tool_result_payload=tool_result,
            requested_chef_name=chef_name,
            scenario="tool_result",
        )
        user_input = f"找不到{chef_name}"
        history = build_history_with_tool_result(f"查一下{chef_name}有没有空", facts)
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = ["H7"]
        capability_tags = ["tool_result", "reply_policy", "specific_chef"]

    else:
        raise ValueError(f"Unknown scenario_key: {scenario_key}")

    # ── Build the generation metadata ─────────────────────────
    gen_meta = {
        "generator": _V02_GENERATOR_META["generator"],
        "model": _V02_GENERATOR_META["model"],
        "seed": seed_i,
        "prompt_sha256": _V02_GENERATOR_META["prompt_sha256"],
        "generated_at": _V02_GENERATOR_META["generated_at"],
        "scenario": scenario_key if not scenario_key.startswith("tool_result") else "tool_result",
        "capability_tags": capability_tags,
        "template_id": "",
    }

    if facts.tool_result_payload is not None:
        candidate_ids = [c.chef_id for c in facts.effective_candidates]
        gen_meta["tool_fact_metadata"] = ToolFactMetadata(
            tool_mode=facts.tool_mode,
            tool_result_status=facts.tool_result_status,
            requested_chef=facts.requested_chef_name,
            candidate_ids=candidate_ids,
            candidate_order=candidate_ids,
            evidence_fields=["chef_id", "chef_name"],
        ).model_dump()

    if scenario_key == "valid_search_tool_call":
        rel_map = {
            "today": "today", "tomorrow": "tomorrow",
            "this_saturday": "this_saturday", "this_sunday": "this_sunday",
        }
        rel_types = list(rel_map.keys())
        rel_type = rel_types[(i - 1) % len(rel_types)]
        gen_meta["relative_time_metadata"] = RelativeTimeMetadata(
            expression_type=rel_type,
            base_datetime=current_time,
            resolved_service_date=facts.booking_state.service_date or "",
            relative_expression=rel_type,
        ).model_dump()

    if scenario_key in ("state_inheritance", "dietary_modification"):
        if scenario_key == "dietary_modification":
            changed = ["dietary_constraints"]
            preserved = ["service_date", "start_time", "people", "address", "cuisine"]
        elif "state_inheritance" in scenario_key:
            mutation_type = (i - 1) % 3
            if mutation_type == 0:
                changed = ["service_date"]
                preserved = ["start_time", "people", "address", "cuisine", "dietary_constraints"]
            elif mutation_type == 1:
                changed = ["cuisine"]
                preserved = ["service_date", "start_time", "people", "address", "dietary_constraints"]
            else:
                changed = []
                preserved = ["service_date", "start_time", "people", "address", "cuisine", "dietary_constraints"]
        else:
            changed = []
            preserved = []
        gen_meta["state_transition_metadata"] = StateTransitionMetadata(
            changed_fields=changed,
            preserved_fields=preserved,
            invalidated_fields=["chef_id", "chef_query_status", "candidate_chefs"] if changed else [],
        ).model_dump()

    inp = {
        "history": history,
        "current_state": current_state,
        "user_input": user_input,
        "current_time": current_time,
        "available_tools": available_tools,
    }

    display_scenario = scenario_key
    if scenario_key.startswith("tool_result"):
        display_scenario = "tool_result"
    elif scenario_key == "dietary_modification":
        display_scenario = "valid_search_tool_call"

    return {
        "id": sid,
        "dataset_version": "phase03_v0.2",
        "contract_id": "homechef-booking-v1",
        "source": "synthetic",
        "scenario": display_scenario,
        "output_kind": output_kind,
        "conversation_kind": conv_kind,
        "tags": capability_tags,
        "input": inp,
        "expected": expected,
        "generation": gen_meta,
        "review": {"status": "machine_validated"},
        "dpo_targets": dpo_targets,
    }


# Full 600 scenario distribution plan
_V02_FULL_PLAN = [
    # missing_required_slots: 60
    *(["missing_required_slots"] * 60),
    # valid_search_tool_call: 100 (includes relative_time)
    *(["valid_search_tool_call"] * 100),
    # state_inheritance: 60
    *(["state_inheritance"] * 60),
    # dietary_modification: 40
    *(["dietary_modification"] * 40),
    # tool_result_matched: 60
    *(["tool_result_matched"] * 60),
    # tool_result_specific_available: 40
    *(["tool_result_specific_available"] * 40),
    # tool_result_specific_unavailable: 40
    *(["tool_result_specific_unavailable"] * 40),
    # tool_result_specific_not_found: 30
    *(["tool_result_specific_not_found"] * 30),
    # tool_result_no_match: 30
    *(["tool_result_no_match"] * 30),
    # tool_result_out_of_service_area: 20
    *(["tool_result_out_of_service_area"] * 20),
    # tool_error: 20
    *(["tool_error"] * 20),
    # explicit_confirmation: 40
    *(["explicit_confirmation"] * 40),
    # rejection: 30
    *(["rejection"] * 30),
    # unrelated: 30
    *(["unrelated"] * 30),
]


def generate_full_raw_v02(output_path: Path, count: int = 600, seed: int = 3001) -> Path:  # noqa: F811
    """Generate v0.2 full raw data (600 rows) using Scenario Lineage Builder.

    Every sample is built from ScenarioFacts via _build_v02_sample(), ensuring
    business lineage consistency across booking_state, tool_call arguments,
    tool_result, and expected Final.
    """
    import random

    rng = random.Random(seed)
    current_time = "2026-08-12 18:00"
    plan = list(_V02_FULL_PLAN[:count])

    lines = []
    for i in range(1, count + 1):
        scenario_key = plan[i - 1]
        sample_dict = _build_v02_sample(i, scenario_key, rng, seed, current_time)
        lines.append(json.dumps(sample_dict, ensure_ascii=False, sort_keys=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ── v0.2.1 Chinese + Difficulty + Diversity Generator ───────────────────────

# Difficulty distribution: Easy=180, Medium=240, Hard=180
_V021_DIFFICULTY_PLAN = (
    ["easy"] * 180 + ["medium"] * 240 + ["hard"] * 180
)


def _build_v02_1_sample(
    i: int,
    scenario_key: str,
    difficulty: str,
    rng,
    seed: int,
    current_time: str = "2026-08-12 18:00",
) -> dict:
    """Build a v0.2.1 sample: v0.2 lineage + Chinese surface + difficulty.

    Wraps _build_v02_sample() and replaces:
    - user_input → Chinese
    - history user/assistant text → Chinese
    - expected.reply → Chinese
    - business entity display names → Chinese
    - Adds difficulty to generation metadata
    """
    import random as _random

    from homechef_booking.data.chinese_realizer import (
        CUISINES,
        DIETARY_COMBOS,
        DIETARY_SINGLE,
        MENU_COMBOS,
        OCCASIONS,
        _pick_chef_display,
        _pick_city_district,
        chinese_reply,
        realize_chinese_user_input,
    )
    from homechef_booking.data.scenario_lineage import ScenarioFacts
    from homechef_booking.schemas.booking import BookingSlot

    # Step 1: Build the base v0.2 sample with deterministic facts
    sample_dict = _build_v02_sample(i, scenario_key, rng, seed, current_time)

    # Step 1.5: Chinese-ify booking_state values FIRST (before building ScenarioFacts)
    local_rng = _random.Random(seed + i + 9999)

    # Chinese-ify current booking_state
    bs_in = sample_dict["input"]["current_state"]["booking_state"]
    city, district = _pick_city_district(local_rng)
    if bs_in.get("address") in [None, "Beijing", "Shanghai", "Hangzhou", "Guangzhou"]:
        bs_in["address"] = f"{city}{district}"
    if bs_in.get("cuisine") in [None, "Sichuan", "Cantonese", "Hunan", "Shandong"]:
        bs_in["cuisine"] = local_rng.choice(CUISINES)
    if bs_in.get("chef_name") and ("Chef" in str(bs_in.get("chef_name", "")) or not any('\u4e00' <= c <= '\u9fff' for c in str(bs_in.get("chef_name", "")))):
        bs_in["chef_name"] = _pick_chef_display(local_rng)

    # Chinese-ify expected booking_state
    bs_exp = sample_dict["expected"].get("booking_state")
    if bs_exp:
        if bs_exp.get("address") in [None, "Beijing", "Shanghai", "Hangzhou", "Guangzhou"]:
            bs_exp["address"] = f"{city}{district}"
        if bs_exp.get("cuisine") in [None, "Sichuan", "Cantonese", "Hunan", "Shandong"]:
            bs_exp["cuisine"] = bs_in.get("cuisine", local_rng.choice(CUISINES))
        if bs_exp.get("chef_name") and ("Chef" in str(bs_exp.get("chef_name", "")) or not any('\u4e00' <= c <= '\u9fff' for c in str(bs_exp.get("chef_name", "")))):
            bs_exp["chef_name"] = bs_in.get("chef_name", _pick_chef_display(local_rng))

    # Step 2: Reconstruct ScenarioFacts from the (now Chinese) sample dict
    bs_dict = sample_dict["input"]["current_state"]["booking_state"]
    bs = BookingSlot.model_validate(bs_dict)

    # Step 2.5: Recompute missing_info/info_complete from Chinese-ified booking_state
    from homechef_booking.schemas.booking import missing_required_slots
    if sample_dict["expected"].get("action") == "final":
        canonical_missing = missing_required_slots(bs)
        sample_dict["expected"]["missing_info"] = list(canonical_missing)
        sample_dict["expected"]["info_complete"] = len(canonical_missing) == 0

    # v0.2.1: Update tool_call arguments in history to match Chinese-ified booking_state
    for _msg in sample_dict["input"]["history"]:
        if _msg.get("role") == "assistant" and _msg.get("tool_calls"):
            for _tc in _msg["tool_calls"]:
                try:
                    _args = json.loads(_tc["function"]["arguments"])
                    if _args.get("address") in [None, "Beijing", "Shanghai", "Hangzhou", "Guangzhou"]:
                        _args["address"] = bs_in.get("address")
                    if _args.get("cuisine") in [None, "Sichuan", "Cantonese", "Hunan", "Shandong"]:
                        _args["cuisine"] = bs_in.get("cuisine")
                    if _args.get("chef_name") and ("Chef" in str(_args.get("chef_name", ""))):
                        _args["chef_name"] = bs_in.get("chef_name")
                    _tc["function"]["arguments"] = json.dumps(_args, ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    pass

    # Determine tool mode and status from history
    history = sample_dict["input"]["history"]
    tool_mode = ""
    tool_result_status = ""
    tool_result_payload = None
    requested_chef = None
    user_confirms = (scenario_key == "explicit_confirmation")
    user_rejects = (scenario_key == "rejection")

    for msg in history:
        role = msg.get("role", "")
        if role == "tool":
            try:
                tr = json.loads(msg["content"])
                tool_mode = tr.get("mode", "")
                tool_result_status = tr.get("status", "")
                tool_result_payload = tr
            except (json.JSONDecodeError, TypeError):
                pass
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                    if args.get("chef_name"):
                        requested_chef = args["chef_name"]
                except (json.JSONDecodeError, TypeError):
                    pass

    # Determine scenario for Chinese realization
    if scenario_key == "dietary_modification":
        realizer_scenario = "dietary_modification"
    elif scenario_key == "valid_search_tool_call" and tool_mode == "specific":
        realizer_scenario = "tool_result_specific_available"
    else:
        realizer_scenario = scenario_key

    facts = ScenarioFacts(
        booking_state=bs,
        tool_mode=tool_mode,
        tool_result_status=tool_result_status,
        tool_result_payload=tool_result_payload,
        requested_chef_name=requested_chef,
        user_confirms=user_confirms,
        user_rejects=user_rejects,
        scenario=realizer_scenario,
    )

    # Step 3: Chinese surface replacement

    # 3a: Replace user_input (allowlist affirmative for confirmation, varied rejection)
    if scenario_key == "explicit_confirmation":
        _AFFIRM = ["确认", "可以", "好的", "就这样", "确认预约"]
        sample_dict["input"]["user_input"] = _AFFIRM[(i - 1) % len(_AFFIRM)]
    elif scenario_key == "rejection":
        _REJECT = ["算了", "先不订了", "取消吧", "不用了", "还是不订了"]
        sample_dict["input"]["user_input"] = _REJECT[(i - 1) % len(_REJECT)]
    else:
        sample_dict["input"]["user_input"] = realize_chinese_user_input(
            scenario_key, facts, local_rng,
        )

    # 3b: Replace history user messages with Chinese
    new_history = []
    for msg in sample_dict["input"]["history"]:
        new_msg = dict(msg)
        if msg.get("role") == "user":
            new_msg["content"] = realize_chinese_user_input(
                scenario_key, facts, local_rng,
            )
        new_history.append(new_msg)
    sample_dict["input"]["history"] = new_history

    # 3c: Replace expected.reply with Chinese
    if sample_dict["expected"].get("action") == "final":
        reply_type = sample_dict["expected"].get("reply_type", "")
        sample_dict["expected"]["reply"] = chinese_reply(reply_type)

    # 3d: Chinese-ify booking_state display names (dietary, occasion)
    bs_in = sample_dict["input"]["current_state"]["booking_state"]
    bs_exp = sample_dict["expected"].get("booking_state", {})

    # Replace dietary_constraints (create singles AND combos for diversity)
    if bs_in.get("dietary_constraints") == ["peanut_allergy"]:
        if local_rng.random() < 0.4:
            bs_in["dietary_constraints"] = [local_rng.choice(DIETARY_SINGLE)]
        else:
            bs_in["dietary_constraints"] = list(local_rng.choice(DIETARY_COMBOS))
    if bs_in.get("dietary_constraints") == ["peanut_allergy", "halal"]:
        bs_in["dietary_constraints"] = list(local_rng.choice(DIETARY_COMBOS))
    if bs_exp.get("dietary_constraints"):
        bs_exp["dietary_constraints"] = bs_in.get("dietary_constraints", [])

    # Replace occasion (consistent between current and expected)
    if bs_in.get("occasion") in [None, "birthday"]:
        new_occasion = local_rng.choice(OCCASIONS)
        bs_in["occasion"] = new_occasion
        if bs_exp is not None:
            bs_exp["occasion"] = new_occasion

    # v0.2.1: Assign menu combinations for menu diversity (always non-empty)
    if not bs_in.get("menu"):
        bs_in["menu"] = list(local_rng.choice(MENU_COMBOS[1:]))  # skip empty combo
    if bs_exp is not None and not bs_exp.get("menu"):
        bs_exp["menu"] = bs_in.get("menu", [])

    # 3e: Replace tool_result candidate chef_names with Chinese
    for msg in sample_dict["input"]["history"]:
        if msg.get("role") == "tool":
            try:
                tr = json.loads(msg["content"])
                for field in ["candidates", "alternatives"]:
                    if field in tr:
                        for c in tr[field]:
                            if "Chef" in str(c.get("chef_name", "")):
                                c["chef_name"] = _pick_chef_display(local_rng)
                if "chef" in tr and isinstance(tr["chef"], dict):
                    if "Chef" in str(tr["chef"].get("chef_name", "")):
                        tr["chef"]["chef_name"] = _pick_chef_display(local_rng)
                msg["content"] = json.dumps(tr, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pass

    # Step 4: Add difficulty
    sample_dict["generation"]["difficulty"] = difficulty

    return sample_dict


def generate_full_raw_v02_1(output_path: Path, count: int = 600, seed: int = 3001) -> Path:
    """Generate v0.2.1 full raw data (600 rows) with Chinese + difficulty + diversity.

    Reuses the v0.2 Scenario Lineage Builder for deterministic business facts,
    then applies Chinese surface realization and difficulty assignment.
    """
    import random as _random

    rng = _random.Random(seed)
    current_time = "2026-08-12 18:00"

    # Shuffle difficulty assignments
    diff_plan = list(_V021_DIFFICULTY_PLAN[:count])
    rng.shuffle(diff_plan)

    # Scenario plan from v0.2
    plan = list(_V02_FULL_PLAN[:count])

    # Relative-time expression type cycle for search_tool_call samples
    from homechef_booking.data.chinese_realizer import (
        RELATIVE_TIME_CHINESE,
        RELATIVE_TIME_RESOLVER,
    )
    rel_types = list(RELATIVE_TIME_RESOLVER.keys())
    rel_idx = 0

    lines = []
    for i in range(1, count + 1):
        scenario_key = plan[i - 1]
        difficulty = diff_plan[i - 1]
        sample_dict = _build_v02_1_sample(i, scenario_key, difficulty, rng, seed, current_time)

        # For search_tool_call / relative_time samples, expand relative-time types
        if scenario_key == "valid_search_tool_call":
            rel_type = rel_types[rel_idx % len(rel_types)]
            rel_idx += 1
            resolved_date = RELATIVE_TIME_RESOLVER[rel_type]
            # Override service_date in current & expected booking_state
            for bs_target in [
                sample_dict["input"]["current_state"]["booking_state"],
                sample_dict["expected"].get("booking_state"),
            ]:
                if bs_target:
                    bs_target["service_date"] = resolved_date
            # Override service_date in expected.arguments (tool_call decision)
            exp_args = sample_dict["expected"].get("arguments")
            if isinstance(exp_args, dict):
                exp_args["service_date"] = resolved_date
            # Override service_date in history assistant tool_call arguments
            for hmsg in sample_dict["input"]["history"]:
                if hmsg.get("role") == "assistant" and hmsg.get("tool_calls"):
                    for htc in hmsg["tool_calls"]:
                        try:
                            hargs = json.loads(htc["function"]["arguments"])
                            hargs["service_date"] = resolved_date
                            htc["function"]["arguments"] = json.dumps(hargs, ensure_ascii=False)
                        except (json.JSONDecodeError, TypeError):
                            pass
            # Override relative_time_metadata
            from homechef_booking.data.raw_sample import RelativeTimeMetadata
            sample_dict["generation"]["relative_time_metadata"] = RelativeTimeMetadata(
                expression_type=rel_type,
                base_datetime=current_time,
                resolved_service_date=resolved_date,
                relative_expression=RELATIVE_TIME_CHINESE.get(rel_type, rel_type),
            ).model_dump()

        lines.append(json.dumps(sample_dict, ensure_ascii=False, sort_keys=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ── v0.2 Lineage-Based Generator ─────────────────────────────


def _v02_complete_booking(rng) -> dict:
    """Build a complete booking_state dict with all required slots filled."""
    return {
        "service_date": f"2026-08-{rng.randint(13, 28):02d}",
        "start_time": f"{rng.randint(11, 20):02d}:00",
        "people": rng.randint(2, 8),
        "address": rng.choice(["Beijing", "Shanghai", "Hangzhou", "Guangzhou"]),
        "cuisine": rng.choice(["Sichuan", "Cantonese", "Hunan", "Shandong"]),
        "budget_min": None,
        "budget_max": None,
        "menu": [],
        "chef_id": None,
        "chef_name": None,
        "ingredient_purchase": None,
        "dietary_constraints": [],
        "occasion": None,
        "confirmation": None,
    }


def _v02_empty_booking() -> dict:
    return {
        "service_date": None, "start_time": None, "people": None, "address": None,
        "cuisine": None, "budget_min": None, "budget_max": None, "menu": [],
        "chef_id": None, "chef_name": None, "ingredient_purchase": None,
        "dietary_constraints": [], "occasion": None, "confirmation": None,
    }


_V02_GENERATOR_META = {
    "generator": "deterministic_smoke_v02",
    "model": "smoke_v02",
    "prompt_sha256": "a" * 64,
    "generated_at": "2026-08-12T00:00:00Z",
}


def generate_smoke_raw_v02(output_path: Path, count: int = 25, seed: int = 3001) -> Path:
    """Generate v0.2 smoke raw data using Scenario Lineage builder.

    Every sample is built from a :class:`ScenarioFacts` object so that
    booking_state, tool_call arguments, tool_result, and expected Final are
    all derived from the same canonical business state.

    The 25-sample plan covers all capability categories:
      1-3:   missing_required_slots (single-slot asks)
      4-7:   valid_search_tool_call (relative time: today/tomorrow/weekend)
      8-10:  valid_search_tool_call (far-future dates)
      11-13: state_inheritance (cuisine change, re-query)
      14:    tool_result / search matched
      15:    tool_result / specific available
      16:    tool_result / specific unavailable
      17:    tool_result / search no_match
      18:    tool_result / search out_of_service_area
      19:    tool_result / search matched (3 candidates, candidate_order)
      20:    dietary modification (state inheritance, re-query)
      21:    explicit_confirmation (booking_authorized)
      22:    rejection (booking_paused)
      23:    state_inheritance (date change, re-query)
      24:    tool_error (booking_paused)
      25:    unrelated (handoff)
    """
    import random

    from homechef_booking.data.scenario_lineage import (
        ScenarioFacts,
        build_current_state,
        build_expected_final,
        build_expected_tool_call,
        build_history_with_tool_result,
    )
    from homechef_booking.data.tool_spec_factory import canonical_find_chefs_tool_dict
    from homechef_booking.schemas.booking import BookingSlot

    rng = random.Random(seed)
    find_chefs_tool = canonical_find_chefs_tool_dict()
    current_time = "2026-08-12 18:00"

    # Deterministic scenario plan for 25 rows
    plan = [
        "missing_required_slots",   # 1
        "missing_required_slots",   # 2
        "missing_required_slots",   # 3
        "valid_search_tool_call",   # 4
        "valid_search_tool_call",   # 5
        "valid_search_tool_call",   # 6
        "valid_search_tool_call",   # 7
        "valid_search_tool_call",   # 8
        "valid_search_tool_call",   # 9
        "valid_search_tool_call",   # 10
        "state_inheritance",        # 11
        "state_inheritance",        # 12
        "state_inheritance",        # 13
        "tool_result_matched",      # 14
        "tool_result_specific_available",  # 15
        "tool_result_specific_unavailable", # 16
        "tool_result_no_match",     # 17
        "tool_result_out_of_service_area", # 18
        "tool_result_matched_3",    # 19
        "dietary_modification",     # 20
        "explicit_confirmation",    # 21
        "rejection",                # 22
        "state_inheritance",        # 23
        "tool_error",               # 24
        "unrelated",                # 25
    ]

    lines = []
    for i in range(1, count + 1):
        sid = f"smoke_v02_{i:03d}"
        scenario_key = plan[i - 1]
        seed_i = seed + i

        # ── Build ScenarioFacts per scenario ──────────────────────
        if scenario_key == "missing_required_slots":
            # Single-slot ask: booking_state has only 1 of 4 required slots
            # The missing_info must list ALL missing slots (canonical)
            which = (i - 1) % 3  # 0,1,2
            bs_dict = _v02_empty_booking()
            user_inputs = [
                "想约一个家宴，麻烦帮我安排",
                "明天晚上需要厨师",
                "需要川菜厨师",
            ]
            user_input = user_inputs[which]
            if which == 0:
                # All 4 required slots missing
                pass
            elif which == 1:
                # Only service_date filled (tomorrow)
                bs_dict["service_date"] = "2026-08-13"
            elif which == 2:
                # Only cuisine filled
                bs_dict["cuisine"] = "Sichuan"
            bs = BookingSlot.model_validate(bs_dict)
            facts = ScenarioFacts(
                booking_state=bs,
                scenario="missing_required_slots",
            )
            history = []
            current_state = build_current_state(facts)
            expected = build_expected_final(facts)
            output_kind = "final"
            conv_kind = "single_turn"
            available_tools = []
            dpo_targets = ["H9"]
            capability_tags = ["missing_required_slots", "reply_policy"]

        elif scenario_key == "valid_search_tool_call":
            # Complete required slots → tool_call
            # Variants 4-7 use relative time, 8-10 use far-future dates
            bs_dict = _v02_complete_booking(rng)
            relative_map = {
                4: ("2026-08-12", "today"),
                5: ("2026-08-13", "tomorrow"),
                6: ("2026-08-15", "this_saturday"),
                7: ("2026-08-16", "this_sunday"),
            }
            if i in relative_map:
                bs_dict["service_date"] = relative_map[i][0]
                rel_type = relative_map[i][1]
            else:
                # Far-future dates
                far_dates = {8: "2026-08-13", 9: "2026-09-01", 10: "2027-01-01"}
                bs_dict["service_date"] = far_dates.get(i, "2026-08-13")
                rel_type = ""
            bs = BookingSlot.model_validate(bs_dict)
            facts = ScenarioFacts(
                booking_state=bs,
                tool_mode="search",
                scenario="valid_search_tool_call",
            )
            history = []
            current_state = build_current_state(facts)
            expected = build_expected_tool_call(facts)
            output_kind = "tool_call"
            conv_kind = "single_turn"
            available_tools = [find_chefs_tool]
            dpo_targets = ["H5"] if i in (4, 5, 6, 7) else ["H2"]
            if i == 6:
                bs_dict["chef_name"] = "Chef Wang"
                facts = ScenarioFacts(
                    booking_state=BookingSlot.model_validate(bs_dict),
                    tool_mode="search",
                    requested_chef_name="Chef Wang",
                    scenario="valid_search_tool_call",
                )
                current_state = build_current_state(facts)
                expected = build_expected_tool_call(facts)
            capability_tags = ["search_tool_call"]
            if i in (4, 5, 6, 7):
                capability_tags.append("relative_time")
            if i == 6:
                capability_tags.append("specific_chef")

        elif scenario_key == "state_inheritance":
            # Multi-turn: current_state has complete booking, user changes cuisine → re-query
            bs_dict = _v02_complete_booking(rng)
            bs_dict["dietary_constraints"] = ["peanut_allergy"]
            if i == 12:
                # User changes cuisine from Sichuan to Cantonese
                bs_dict["cuisine"] = "Sichuan"
                user_input = "改成粤菜"
            elif i == 13:
                bs_dict["cuisine"] = "Sichuan"
                user_input = "再查一次"
            else:
                # i == 11: user changes date
                bs_dict["service_date"] = "2026-08-13"
                user_input = "改成8月14号"
            bs = BookingSlot.model_validate(bs_dict)
            # Expected: tool_call with new cuisine/date
            expected_bs_dict = dict(bs_dict)
            if i == 12:
                expected_bs_dict["cuisine"] = "Cantonese"
            elif i == 11:
                expected_bs_dict["service_date"] = "2026-08-14"
            elif i == 23:
                expected_bs_dict["service_date"] = "2026-08-14"
            expected_bs = BookingSlot.model_validate(expected_bs_dict)
            facts = ScenarioFacts(
                booking_state=expected_bs,
                tool_mode="search",
                scenario="state_inheritance",
            )
            history = []
            current_state = {
                "booking_state": bs_dict,
                "chef_query_status": "not_checked",
                "candidate_chefs": [],
                "awaiting_confirmation": False,
            }
            expected = build_expected_tool_call(facts)
            output_kind = "tool_call"
            conv_kind = "multi_turn"
            available_tools = [find_chefs_tool]
            dpo_targets = ["H3", "H4"]
            capability_tags = ["state_inheritance", "modification_requires_requery"]

        elif scenario_key == "tool_result_matched":
            # search/matched → present_chef_candidates
            bs_dict = _v02_complete_booking(rng)
            bs = BookingSlot.model_validate(bs_dict)
            candidates = [
                {"chef_id": "C003", "chef_name": "Chef Wang"},
                {"chef_id": "C007", "chef_name": "Chef Li"},
            ]
            tool_result = {"mode": "search", "status": "matched", "candidates": candidates}
            facts = ScenarioFacts(
                booking_state=bs,
                tool_mode="search",
                tool_result_status="matched",
                tool_result_payload=tool_result,
                scenario="tool_result",
            )
            user_input = "找到2位厨师"
            history = build_history_with_tool_result("find chefs", facts)
            current_state = build_current_state(facts)
            expected = build_expected_final(facts)
            output_kind = "final"
            conv_kind = "multi_turn"
            available_tools = []
            dpo_targets = ["H1", "H7", "H8"]
            capability_tags = ["tool_result", "tool_fact_grounding", "reply_policy"]

        elif scenario_key == "tool_result_specific_available":
            # specific/available → confirm_specific_chef
            bs_dict = _v02_complete_booking(rng)
            bs_dict["chef_name"] = "Chef Zhang"
            bs = BookingSlot.model_validate(bs_dict)
            chef = {"chef_id": "C005", "chef_name": "Chef Zhang"}
            tool_result = {"mode": "specific", "status": "available", "chef": chef}
            facts = ScenarioFacts(
                booking_state=bs,
                tool_mode="specific",
                tool_result_status="available",
                tool_result_payload=tool_result,
                requested_chef_name="Chef Zhang",
                scenario="tool_result",
            )
            user_input = "Chef Zhang有档期"
            history = build_history_with_tool_result("查一下Chef Zhang有没有空", facts)
            current_state = build_current_state(facts)
            expected = build_expected_final(facts)
            output_kind = "final"
            conv_kind = "multi_turn"
            available_tools = []
            dpo_targets = ["H1", "H7", "H8"]
            capability_tags = ["tool_result", "tool_fact_grounding", "reply_policy", "specific_chef"]

        elif scenario_key == "tool_result_specific_unavailable":
            # specific/unavailable → present_alternatives
            bs_dict = _v02_complete_booking(rng)
            bs_dict["chef_name"] = "Chef Zhang"
            bs = BookingSlot.model_validate(bs_dict)
            alternatives = [
                {"chef_id": "C006", "chef_name": "Chef Chen"},
                {"chef_id": "C008", "chef_name": "Chef Liu"},
            ]
            tool_result = {
                "mode": "specific", "status": "unavailable",
                "requested_chef": "Chef Zhang", "alternatives": alternatives,
            }
            facts = ScenarioFacts(
                booking_state=bs,
                tool_mode="specific",
                tool_result_status="unavailable",
                tool_result_payload=tool_result,
                requested_chef_name="Chef Zhang",
                scenario="tool_result",
            )
            user_input = "Chef Zhang无档期，有哪些替代"
            history = build_history_with_tool_result("查一下Chef Zhang有没有空", facts)
            current_state = build_current_state(facts)
            expected = build_expected_final(facts)
            output_kind = "final"
            conv_kind = "multi_turn"
            available_tools = []
            dpo_targets = ["H7", "H8"]
            capability_tags = ["tool_result", "tool_fact_grounding", "reply_policy", "specific_chef"]

        elif scenario_key == "tool_result_no_match":
            # search/no_match → inform_no_match
            bs_dict = _v02_complete_booking(rng)
            bs = BookingSlot.model_validate(bs_dict)
            tool_result = {"mode": "search", "status": "no_match", "candidates": []}
            facts = ScenarioFacts(
                booking_state=bs,
                tool_mode="search",
                tool_result_status="no_match",
                tool_result_payload=tool_result,
                scenario="tool_result",
            )
            user_input = "没找到合适的厨师吗"
            history = build_history_with_tool_result("find chefs", facts)
            current_state = build_current_state(facts)
            expected = build_expected_final(facts)
            output_kind = "final"
            conv_kind = "multi_turn"
            available_tools = []
            dpo_targets = ["H4"]
            capability_tags = ["tool_result", "reply_policy"]

        elif scenario_key == "tool_result_out_of_service_area":
            # search/out_of_service_area → inform_out_of_service_area
            bs_dict = _v02_complete_booking(rng)
            bs = BookingSlot.model_validate(bs_dict)
            tool_result = {"mode": "search", "status": "out_of_service_area", "candidates": []}
            facts = ScenarioFacts(
                booking_state=bs,
                tool_mode="search",
                tool_result_status="out_of_service_area",
                tool_result_payload=tool_result,
                scenario="tool_result",
            )
            user_input = "这个地址能服务吗"
            history = build_history_with_tool_result("find chefs", facts)
            current_state = build_current_state(facts)
            expected = build_expected_final(facts)
            output_kind = "final"
            conv_kind = "multi_turn"
            available_tools = []
            dpo_targets = []
            capability_tags = ["tool_result", "reply_policy"]

        elif scenario_key == "tool_result_matched_3":
            # search/matched with 3 candidates → present_chef_candidates (candidate_order)
            bs_dict = _v02_complete_booking(rng)
            bs = BookingSlot.model_validate(bs_dict)
            candidates = [
                {"chef_id": "C003", "chef_name": "Chef Wang"},
                {"chef_id": "C007", "chef_name": "Chef Li"},
                {"chef_id": "C011", "chef_name": "Chef Zhao"},
            ]
            tool_result = {"mode": "search", "status": "matched", "candidates": candidates}
            facts = ScenarioFacts(
                booking_state=bs,
                tool_mode="search",
                tool_result_status="matched",
                tool_result_payload=tool_result,
                scenario="tool_result",
            )
            user_input = "找到3位厨师"
            history = build_history_with_tool_result("find chefs", facts)
            current_state = build_current_state(facts)
            expected = build_expected_final(facts)
            output_kind = "final"
            conv_kind = "multi_turn"
            available_tools = []
            dpo_targets = ["H8"]
            capability_tags = ["tool_result", "candidate_order", "reply_policy"]

        elif scenario_key == "dietary_modification":
            # User modifies dietary_constraints → re-query required
            # current_state has complete booking with existing dietary
            bs_dict = _v02_complete_booking(rng)
            bs_dict["dietary_constraints"] = ["peanut_allergy"]
            bs = BookingSlot.model_validate(bs_dict)
            # Expected: tool_call with new dietary_constraints added
            expected_bs_dict = dict(bs_dict)
            expected_bs_dict["dietary_constraints"] = ["peanut_allergy", "halal"]
            expected_bs = BookingSlot.model_validate(expected_bs_dict)
            facts = ScenarioFacts(
                booking_state=expected_bs,
                tool_mode="search",
                scenario="valid_search_tool_call",
            )
            user_input = "找粤菜厨师，花生过敏，清真"
            history = []
            current_state = {
                "booking_state": bs_dict,
                "chef_query_status": "not_checked",
                "candidate_chefs": [],
                "awaiting_confirmation": False,
            }
            expected = build_expected_tool_call(facts)
            output_kind = "tool_call"
            conv_kind = "multi_turn"
            available_tools = [find_chefs_tool]
            dpo_targets = ["H4"]
            capability_tags = ["dietary_preservation", "modification_requires_requery", "state_inheritance"]

        elif scenario_key == "explicit_confirmation":
            # awaiting_confirmation=True, user confirms → booking_authorized
            bs_dict = _v02_complete_booking(rng)
            bs_dict["chef_id"] = "C005"
            bs_dict["chef_name"] = "Chef Zhang"
            bs = BookingSlot.model_validate(bs_dict)
            chef = {"chef_id": "C005", "chef_name": "Chef Zhang"}
            tool_result = {"mode": "specific", "status": "available", "chef": chef}
            facts = ScenarioFacts(
                booking_state=bs,
                tool_mode="specific",
                tool_result_status="available",
                tool_result_payload=tool_result,
                requested_chef_name="Chef Zhang",
                user_confirms=True,
                scenario="explicit_confirmation",
            )
            user_input = "确认"
            history = []
            current_state = build_current_state(facts)
            expected = build_expected_final(facts)
            output_kind = "final"
            conv_kind = "multi_turn"
            available_tools = []
            dpo_targets = ["H6"]
            capability_tags = ["confirmation", "reply_policy"]

        elif scenario_key == "rejection":
            # awaiting_confirmation=True, user declines → booking_paused
            bs_dict = _v02_complete_booking(rng)
            bs_dict["chef_id"] = "C005"
            bs_dict["chef_name"] = "Chef Zhang"
            bs = BookingSlot.model_validate(bs_dict)
            chef = {"chef_id": "C005", "chef_name": "Chef Zhang"}
            tool_result = {"mode": "specific", "status": "available", "chef": chef}
            facts = ScenarioFacts(
                booking_state=bs,
                tool_mode="specific",
                tool_result_status="available",
                tool_result_payload=tool_result,
                requested_chef_name="Chef Zhang",
                user_rejects=True,
                scenario="rejection",
            )
            user_input = "算了"
            history = []
            current_state = build_current_state(facts)
            expected = build_expected_final(facts)
            output_kind = "final"
            conv_kind = "multi_turn"
            available_tools = []
            dpo_targets = ["H6"]
            capability_tags = ["confirmation", "reply_policy"]

        elif scenario_key == "tool_error":
            # search/error → booking_paused
            bs_dict = _v02_complete_booking(rng)
            bs = BookingSlot.model_validate(bs_dict)
            tool_result = {
                "mode": "search", "status": "error",
                "error_code": "SERVICE_ERROR", "retryable": False,
                "message": "查询服务暂时不可用",
            }
            facts = ScenarioFacts(
                booking_state=bs,
                tool_mode="search",
                tool_result_status="error",
                tool_result_payload=tool_result,
                scenario="tool_error",
            )
            user_input = "查询出错了"
            history = build_history_with_tool_result("find chefs", facts)
            current_state = build_current_state(facts)
            expected = build_expected_final(facts)
            output_kind = "final"
            conv_kind = "multi_turn"
            available_tools = []
            dpo_targets = []
            capability_tags = ["tool_result", "reply_policy"]

        elif scenario_key == "unrelated":
            # Non-booking question → handoff
            bs_dict = _v02_empty_booking()
            bs = BookingSlot.model_validate(bs_dict)
            facts = ScenarioFacts(
                booking_state=bs,
                scenario="unrelated",
            )
            user_input = "今天天气怎么样"
            history = []
            current_state = build_current_state(facts)
            expected = build_expected_final(facts)
            output_kind = "final"
            conv_kind = "single_turn"
            available_tools = []
            dpo_targets = []
            capability_tags = ["reply_policy"]

        else:
            raise ValueError(f"Unknown scenario_key: {scenario_key}")

        # ── Build the generation metadata ─────────────────────────
        from homechef_booking.data.raw_sample import (
            RelativeTimeMetadata,
            ToolFactMetadata,
        )

        gen_meta = {
            "generator": _V02_GENERATOR_META["generator"],
            "model": _V02_GENERATOR_META["model"],
            "seed": seed_i,
            "prompt_sha256": _V02_GENERATOR_META["prompt_sha256"],
            "generated_at": _V02_GENERATOR_META["generated_at"],
            "scenario": scenario_key if not scenario_key.startswith("tool_result") else "tool_result",
            "capability_tags": capability_tags,
            "template_id": "",
        }

        # Add ToolFactMetadata for tool_result samples
        if facts.tool_result_payload is not None:
            candidate_ids = [c.chef_id for c in facts.effective_candidates]
            gen_meta["tool_fact_metadata"] = ToolFactMetadata(
                tool_mode=facts.tool_mode,
                tool_result_status=facts.tool_result_status,
                requested_chef=facts.requested_chef_name,
                candidate_ids=candidate_ids,
                candidate_order=candidate_ids,
                evidence_fields=["chef_id", "chef_name"],
            ).model_dump()

        # Add RelativeTimeMetadata for relative time samples
        if scenario_key == "valid_search_tool_call" and i in (4, 5, 6, 7):
            rel_map = {4: "today", 5: "tomorrow", 6: "this_saturday", 7: "this_sunday"}
            rel_type = rel_map[i]
            gen_meta["relative_time_metadata"] = RelativeTimeMetadata(
                expression_type=rel_type,
                base_datetime=current_time,
                resolved_service_date=facts.booking_state.service_date or "",
                relative_expression=rel_type,
            ).model_dump()

        # Add StateTransitionMetadata for state_inheritance / dietary_modification
        if scenario_key in ("state_inheritance", "dietary_modification"):
            from homechef_booking.data.raw_sample import StateTransitionMetadata
            if scenario_key == "dietary_modification":
                changed = ["dietary_constraints"]
                preserved = ["service_date", "start_time", "people", "address", "cuisine"]
            elif i == 11 or i == 23:
                changed = ["service_date"]
                preserved = ["start_time", "people", "address", "cuisine", "dietary_constraints"]
            elif i == 12:
                changed = ["cuisine"]
                preserved = ["service_date", "start_time", "people", "address", "dietary_constraints"]
            else:
                changed = []
                preserved = ["service_date", "start_time", "people", "address", "cuisine", "dietary_constraints"]
            gen_meta["state_transition_metadata"] = StateTransitionMetadata(
                changed_fields=changed,
                preserved_fields=preserved,
                invalidated_fields=["chef_id", "chef_query_status", "candidate_chefs"] if changed else [],
            ).model_dump()

        # Build input dict
        inp = {
            "history": history,
            "current_state": current_state,
            "user_input": user_input,
            "current_time": current_time,
            "available_tools": available_tools,
        }

        # Map scenario_key to display scenario
        display_scenario = scenario_key
        if scenario_key.startswith("tool_result"):
            display_scenario = "tool_result"
        elif scenario_key == "dietary_modification":
            display_scenario = "valid_search_tool_call"

        lines.append(json.dumps({
            "id": sid,
            "dataset_version": "phase03_v0.2",
            "contract_id": "homechef-booking-v1",
            "source": "synthetic",
            "scenario": display_scenario,
            "output_kind": output_kind,
            "conversation_kind": conv_kind,
            "tags": capability_tags,
            "input": inp,
            "expected": expected,
            "generation": gen_meta,
            "review": {"status": "machine_validated"},
            "dpo_targets": dpo_targets,
        }, ensure_ascii=False, sort_keys=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ── v0.2 Full Generator (600 rows, reuses Scenario Lineage Builder) ────────


def _build_v02_sample(
    i: int,
    scenario_key: str,
    rng,
    seed: int,
    current_time: str = "2026-08-12 18:00",
) -> dict:
    """Build a single v0.2 raw sample dict from ScenarioFacts lineage.

    This is the shared builder used by both smoke (25) and full (600)
    generators.  Every sample is derived from a ScenarioFacts object.
    """
    from homechef_booking.data.raw_sample import (
        RelativeTimeMetadata,
        StateTransitionMetadata,
        ToolFactMetadata,
    )
    from homechef_booking.data.scenario_lineage import (
        ScenarioFacts,
        build_current_state,
        build_expected_final,
        build_expected_tool_call,
        build_history_with_tool_result,
    )
    from homechef_booking.data.tool_spec_factory import canonical_find_chefs_tool_dict
    from homechef_booking.schemas.booking import BookingSlot

    find_chefs_tool = canonical_find_chefs_tool_dict()
    sid = f"phase03-raw-v02-{i:06d}"
    seed_i = seed + i

    if scenario_key == "missing_required_slots":
        which = (i - 1) % 3
        bs_dict = _v02_empty_booking()
        user_inputs = [
            "想约一个家宴，麻烦帮我安排",
            "明天晚上需要厨师",
            "需要川菜厨师",
        ]
        user_input = user_inputs[which]
        if which == 1:
            bs_dict["service_date"] = "2026-08-13"
        elif which == 2:
            bs_dict["cuisine"] = "Sichuan"
        bs = BookingSlot.model_validate(bs_dict)
        facts = ScenarioFacts(booking_state=bs, scenario="missing_required_slots")
        history = []
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "single_turn"
        available_tools = []
        dpo_targets = ["H2"]
        capability_tags = ["missing_required_slots", "reply_policy"]

    elif scenario_key == "valid_search_tool_call":
        bs_dict = _v02_complete_booking(rng)
        relative_map = {
            "today": ("2026-08-12", "today"),
            "tomorrow": ("2026-08-13", "tomorrow"),
            "this_saturday": ("2026-08-15", "this_saturday"),
            "this_sunday": ("2026-08-16", "this_sunday"),
        }
        rel_types = list(relative_map.keys())
        rel_idx = (i - 1) % len(rel_types)
        rel_key = rel_types[rel_idx]
        bs_dict["service_date"] = relative_map[rel_key][0]
        rel_type = rel_key
        bs = BookingSlot.model_validate(bs_dict)
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="search",
            scenario="valid_search_tool_call",
        )
        history = []
        current_state = build_current_state(facts)
        expected = build_expected_tool_call(facts)
        output_kind = "tool_call"
        conv_kind = "single_turn"
        available_tools = [find_chefs_tool]
        dpo_targets = ["H5"]
        capability_tags = ["search_tool_call", "relative_time"]
        user_input = f"需要{bs.cuisine}厨师，{bs.service_date}，{bs.people}人，{bs.address}"

    elif scenario_key == "state_inheritance":
        bs_dict = _v02_complete_booking(rng)
        bs_dict["dietary_constraints"] = ["peanut_allergy"]
        mutation_type = (i - 1) % 3  # 0=date, 1=cuisine, 2=re-query
        if mutation_type == 1:
            bs_dict["cuisine"] = "Sichuan"
            user_input = "改成粤菜"
        elif mutation_type == 2:
            bs_dict["cuisine"] = "Sichuan"
            user_input = "再查一次"
        else:
            bs_dict["service_date"] = "2026-08-13"
            user_input = "改成8月14号"
        bs = BookingSlot.model_validate(bs_dict)
        expected_bs_dict = dict(bs_dict)
        if mutation_type == 1:
            expected_bs_dict["cuisine"] = "Cantonese"
        elif mutation_type == 0:
            expected_bs_dict["service_date"] = "2026-08-14"
        expected_bs = BookingSlot.model_validate(expected_bs_dict)
        facts = ScenarioFacts(
            booking_state=expected_bs,
            tool_mode="search",
            scenario="state_inheritance",
        )
        history = []
        current_state = {
            "booking_state": bs_dict,
            "chef_query_status": "not_checked",
            "candidate_chefs": [],
            "awaiting_confirmation": False,
        }
        expected = build_expected_tool_call(facts)
        output_kind = "tool_call"
        conv_kind = "multi_turn"
        available_tools = [find_chefs_tool]
        dpo_targets = ["H3", "H4"]
        capability_tags = ["state_inheritance", "modification_requires_requery"]

    elif scenario_key == "tool_result_matched":
        bs_dict = _v02_complete_booking(rng)
        bs = BookingSlot.model_validate(bs_dict)
        n_cands = 2 + (i % 3)  # 2-4 candidates
        chef_names = ["Chef Wang", "Chef Li", "Chef Zhao", "Chef Chen", "Chef Liu"]
        candidates = [
            {"chef_id": f"C{i:03d}", "chef_name": chef_names[j % len(chef_names)]}
            for j, i in enumerate(range(3, 3 + n_cands), 3)
        ]
        tool_result = {"mode": "search", "status": "matched", "candidates": candidates}
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="search",
            tool_result_status="matched",
            tool_result_payload=tool_result,
            scenario="tool_result",
        )
        user_input = f"找到{len(candidates)}位厨师"
        history = build_history_with_tool_result("find chefs", facts)
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = ["H1", "H7", "H8"]
        capability_tags = ["tool_result", "tool_fact_grounding", "reply_policy", "candidate_order"]

    elif scenario_key == "tool_result_specific_available":
        bs_dict = _v02_complete_booking(rng)
        chef_name = rng.choice(["Chef Zhang", "Chef Wang", "Chef Li"])
        bs_dict["chef_name"] = chef_name
        bs = BookingSlot.model_validate(bs_dict)
        chef_id = f"C{rng.randint(100, 999)}"
        chef = {"chef_id": chef_id, "chef_name": chef_name}
        tool_result = {"mode": "specific", "status": "available", "chef": chef}
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="specific",
            tool_result_status="available",
            tool_result_payload=tool_result,
            requested_chef_name=chef_name,
            scenario="tool_result",
        )
        user_input = f"{chef_name}有档期"
        history = build_history_with_tool_result(f"查一下{chef_name}有没有空", facts)
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = ["H1", "H7", "H8"]
        capability_tags = ["tool_result", "tool_fact_grounding", "reply_policy", "specific_chef"]

    elif scenario_key == "tool_result_specific_unavailable":
        bs_dict = _v02_complete_booking(rng)
        chef_name = rng.choice(["Chef Zhang", "Chef Wang", "Chef Li"])
        bs_dict["chef_name"] = chef_name
        bs = BookingSlot.model_validate(bs_dict)
        alt_names = ["Chef Chen", "Chef Liu", "Chef Zhao"]
        n_alts = 1 + (i % 2)
        alternatives = [
            {"chef_id": f"C{rng.randint(200, 899)}", "chef_name": alt_names[j % len(alt_names)]}
            for j in range(n_alts)
        ]
        tool_result = {
            "mode": "specific", "status": "unavailable",
            "requested_chef": chef_name, "alternatives": alternatives,
        }
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="specific",
            tool_result_status="unavailable",
            tool_result_payload=tool_result,
            requested_chef_name=chef_name,
            scenario="tool_result",
        )
        user_input = f"{chef_name}无档期，有哪些替代"
        history = build_history_with_tool_result(f"查一下{chef_name}有没有空", facts)
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = ["H7", "H8"]
        capability_tags = ["tool_result", "tool_fact_grounding", "reply_policy", "specific_chef"]

    elif scenario_key == "tool_result_no_match":
        bs_dict = _v02_complete_booking(rng)
        bs = BookingSlot.model_validate(bs_dict)
        tool_result = {"mode": "search", "status": "no_match", "candidates": []}
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="search",
            tool_result_status="no_match",
            tool_result_payload=tool_result,
            scenario="tool_result",
        )
        user_input = "没找到合适的厨师吗"
        history = build_history_with_tool_result("find chefs", facts)
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = ["H4"]
        capability_tags = ["tool_result", "reply_policy"]

    elif scenario_key == "tool_result_out_of_service_area":
        bs_dict = _v02_complete_booking(rng)
        bs = BookingSlot.model_validate(bs_dict)
        tool_result = {"mode": "search", "status": "out_of_service_area", "candidates": []}
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="search",
            tool_result_status="out_of_service_area",
            tool_result_payload=tool_result,
            scenario="tool_result",
        )
        user_input = "这个地址能服务吗"
        history = build_history_with_tool_result("find chefs", facts)
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = []
        capability_tags = ["tool_result", "reply_policy"]

    elif scenario_key == "dietary_modification":
        bs_dict = _v02_complete_booking(rng)
        bs_dict["dietary_constraints"] = ["peanut_allergy"]
        bs = BookingSlot.model_validate(bs_dict)
        expected_bs_dict = dict(bs_dict)
        expected_bs_dict["dietary_constraints"] = ["peanut_allergy", "halal"]
        expected_bs = BookingSlot.model_validate(expected_bs_dict)
        facts = ScenarioFacts(
            booking_state=expected_bs,
            tool_mode="search",
            scenario="valid_search_tool_call",
        )
        user_input = "找粤菜厨师，花生过敏，清真"
        history = []
        current_state = {
            "booking_state": bs_dict,
            "chef_query_status": "not_checked",
            "candidate_chefs": [],
            "awaiting_confirmation": False,
        }
        expected = build_expected_tool_call(facts)
        output_kind = "tool_call"
        conv_kind = "multi_turn"
        available_tools = [find_chefs_tool]
        dpo_targets = ["H4"]
        capability_tags = ["dietary_preservation", "modification_requires_requery", "state_inheritance"]

    elif scenario_key == "explicit_confirmation":
        bs_dict = _v02_complete_booking(rng)
        chef_name = rng.choice(["Chef Zhang", "Chef Wang"])
        chef_id = f"C{rng.randint(100, 999)}"
        bs_dict["chef_id"] = chef_id
        bs_dict["chef_name"] = chef_name
        bs = BookingSlot.model_validate(bs_dict)
        chef = {"chef_id": chef_id, "chef_name": chef_name}
        tool_result = {"mode": "specific", "status": "available", "chef": chef}
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="specific",
            tool_result_status="available",
            tool_result_payload=tool_result,
            requested_chef_name=chef_name,
            user_confirms=True,
            scenario="explicit_confirmation",
        )
        user_input = "确认"
        history = []
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = ["H6"]
        capability_tags = ["confirmation", "reply_policy"]

    elif scenario_key == "rejection":
        bs_dict = _v02_complete_booking(rng)
        chef_name = rng.choice(["Chef Zhang", "Chef Wang"])
        chef_id = f"C{rng.randint(100, 999)}"
        bs_dict["chef_id"] = chef_id
        bs_dict["chef_name"] = chef_name
        bs = BookingSlot.model_validate(bs_dict)
        chef = {"chef_id": chef_id, "chef_name": chef_name}
        tool_result = {"mode": "specific", "status": "available", "chef": chef}
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="specific",
            tool_result_status="available",
            tool_result_payload=tool_result,
            requested_chef_name=chef_name,
            user_rejects=True,
            scenario="rejection",
        )
        user_input = "算了"
        history = []
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = ["H6"]
        capability_tags = ["confirmation", "reply_policy"]

    elif scenario_key == "tool_error":
        bs_dict = _v02_complete_booking(rng)
        bs = BookingSlot.model_validate(bs_dict)
        tool_result = {
            "mode": "search", "status": "error",
            "error_code": "SERVICE_ERROR", "retryable": False,
            "message": "查询服务暂时不可用",
        }
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="search",
            tool_result_status="error",
            tool_result_payload=tool_result,
            scenario="tool_error",
        )
        user_input = "查询出错了"
        history = build_history_with_tool_result("find chefs", facts)
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = []
        capability_tags = ["tool_result", "reply_policy"]

    elif scenario_key == "unrelated":
        bs_dict = _v02_empty_booking()
        bs = BookingSlot.model_validate(bs_dict)
        facts = ScenarioFacts(booking_state=bs, scenario="unrelated")
        user_inputs = ["今天天气怎么样", "帮我推荐一部电影", "明天股市行情如何"]
        user_input = user_inputs[i % len(user_inputs)]
        history = []
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "single_turn"
        available_tools = []
        dpo_targets = []
        capability_tags = ["reply_policy"]

    elif scenario_key == "tool_result_specific_not_found":
        bs_dict = _v02_complete_booking(rng)
        chef_name = f"Chef Ghost{i}"
        bs_dict["chef_name"] = chef_name
        bs = BookingSlot.model_validate(bs_dict)
        tool_result = {
            "mode": "specific", "status": "not_found",
            "requested_chef": chef_name, "alternatives": [],
        }
        facts = ScenarioFacts(
            booking_state=bs,
            tool_mode="specific",
            tool_result_status="not_found",
            tool_result_payload=tool_result,
            requested_chef_name=chef_name,
            scenario="tool_result",
        )
        user_input = f"找不到{chef_name}"
        history = build_history_with_tool_result(f"查一下{chef_name}有没有空", facts)
        current_state = build_current_state(facts)
        expected = build_expected_final(facts)
        output_kind = "final"
        conv_kind = "multi_turn"
        available_tools = []
        dpo_targets = ["H7"]
        capability_tags = ["tool_result", "reply_policy", "specific_chef"]

    else:
        raise ValueError(f"Unknown scenario_key: {scenario_key}")

    # ── Build the generation metadata ─────────────────────────
    gen_meta = {
        "generator": _V02_GENERATOR_META["generator"],
        "model": _V02_GENERATOR_META["model"],
        "seed": seed_i,
        "prompt_sha256": _V02_GENERATOR_META["prompt_sha256"],
        "generated_at": _V02_GENERATOR_META["generated_at"],
        "scenario": scenario_key if not scenario_key.startswith("tool_result") else "tool_result",
        "capability_tags": capability_tags,
        "template_id": "",
    }

    if facts.tool_result_payload is not None:
        candidate_ids = [c.chef_id for c in facts.effective_candidates]
        gen_meta["tool_fact_metadata"] = ToolFactMetadata(
            tool_mode=facts.tool_mode,
            tool_result_status=facts.tool_result_status,
            requested_chef=facts.requested_chef_name,
            candidate_ids=candidate_ids,
            candidate_order=candidate_ids,
            evidence_fields=["chef_id", "chef_name"],
        ).model_dump()

    if scenario_key == "valid_search_tool_call":
        rel_map = {
            "today": "today", "tomorrow": "tomorrow",
            "this_saturday": "this_saturday", "this_sunday": "this_sunday",
        }
        rel_types = list(rel_map.keys())
        rel_type = rel_types[(i - 1) % len(rel_types)]
        gen_meta["relative_time_metadata"] = RelativeTimeMetadata(
            expression_type=rel_type,
            base_datetime=current_time,
            resolved_service_date=facts.booking_state.service_date or "",
            relative_expression=rel_type,
        ).model_dump()

    if scenario_key in ("state_inheritance", "dietary_modification"):
        if scenario_key == "dietary_modification":
            changed = ["dietary_constraints"]
            preserved = ["service_date", "start_time", "people", "address", "cuisine"]
        elif "state_inheritance" in scenario_key:
            mutation_type = (i - 1) % 3
            if mutation_type == 0:
                changed = ["service_date"]
                preserved = ["start_time", "people", "address", "cuisine", "dietary_constraints"]
            elif mutation_type == 1:
                changed = ["cuisine"]
                preserved = ["service_date", "start_time", "people", "address", "dietary_constraints"]
            else:
                changed = []
                preserved = ["service_date", "start_time", "people", "address", "cuisine", "dietary_constraints"]
        else:
            changed = []
            preserved = []
        gen_meta["state_transition_metadata"] = StateTransitionMetadata(
            changed_fields=changed,
            preserved_fields=preserved,
            invalidated_fields=["chef_id", "chef_query_status", "candidate_chefs"] if changed else [],
        ).model_dump()

    inp = {
        "history": history,
        "current_state": current_state,
        "user_input": user_input,
        "current_time": current_time,
        "available_tools": available_tools,
    }

    display_scenario = scenario_key
    if scenario_key.startswith("tool_result"):
        display_scenario = "tool_result"
    elif scenario_key == "dietary_modification":
        display_scenario = "valid_search_tool_call"

    return {
        "id": sid,
        "dataset_version": "phase03_v0.2",
        "contract_id": "homechef-booking-v1",
        "source": "synthetic",
        "scenario": display_scenario,
        "output_kind": output_kind,
        "conversation_kind": conv_kind,
        "tags": capability_tags,
        "input": inp,
        "expected": expected,
        "generation": gen_meta,
        "review": {"status": "machine_validated"},
        "dpo_targets": dpo_targets,
    }


# Full 600 scenario distribution plan
_V02_FULL_PLAN = [
    # missing_required_slots: 60
    *(["missing_required_slots"] * 60),
    # valid_search_tool_call: 100 (includes relative_time)
    *(["valid_search_tool_call"] * 100),
    # state_inheritance: 60
    *(["state_inheritance"] * 60),
    # dietary_modification: 40
    *(["dietary_modification"] * 40),
    # tool_result_matched: 60
    *(["tool_result_matched"] * 60),
    # tool_result_specific_available: 40
    *(["tool_result_specific_available"] * 40),
    # tool_result_specific_unavailable: 40
    *(["tool_result_specific_unavailable"] * 40),
    # tool_result_specific_not_found: 30
    *(["tool_result_specific_not_found"] * 30),
    # tool_result_no_match: 30
    *(["tool_result_no_match"] * 30),
    # tool_result_out_of_service_area: 20
    *(["tool_result_out_of_service_area"] * 20),
    # tool_error: 20
    *(["tool_error"] * 20),
    # explicit_confirmation: 40
    *(["explicit_confirmation"] * 40),
    # rejection: 30
    *(["rejection"] * 30),
    # unrelated: 30
    *(["unrelated"] * 30),
]


def generate_full_raw_v02(output_path: Path, count: int = 600, seed: int = 3001) -> Path:
    """Generate v0.2 full raw data (600 rows) using Scenario Lineage Builder.

    Every sample is built from ScenarioFacts via _build_v02_sample(), ensuring
    business lineage consistency across booking_state, tool_call arguments,
    tool_result, and expected Final.
    """
    import random

    rng = random.Random(seed)
    current_time = "2026-08-12 18:00"
    plan = list(_V02_FULL_PLAN[:count])

    lines = []
    for i in range(1, count + 1):
        scenario_key = plan[i - 1]
        sample_dict = _build_v02_sample(i, scenario_key, rng, seed, current_time)
        lines.append(json.dumps(sample_dict, ensure_ascii=False, sort_keys=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ── v0.2.1 Chinese + Difficulty + Diversity Generator ───────────────────────

# Difficulty distribution: Easy=180, Medium=240, Hard=180
_V021_DIFFICULTY_PLAN = (
    ["easy"] * 180 + ["medium"] * 240 + ["hard"] * 180
)


def _build_v02_1_sample(
    i: int,
    scenario_key: str,
    difficulty: str,
    rng,
    seed: int,
    current_time: str = "2026-08-12 18:00",
) -> dict:
    """Build a v0.2.1 sample: v0.2 lineage + Chinese surface + difficulty.

    Wraps _build_v02_sample() and replaces:
    - user_input → Chinese
    - history user/assistant text → Chinese
    - expected.reply → Chinese
    - business entity display names → Chinese
    - Adds difficulty to generation metadata
    """
    import random as _random

    from homechef_booking.data.chinese_realizer import (
        CUISINES,
        DIETARY_COMBOS,
        DIETARY_SINGLE,
        MENU_COMBOS,
        OCCASIONS,
        _pick_chef_display,
        _pick_city_district,
        chinese_reply,
        realize_chinese_user_input,
    )
    from homechef_booking.data.scenario_lineage import ScenarioFacts
    from homechef_booking.schemas.booking import BookingSlot

    # Step 1: Build the base v0.2 sample with deterministic facts
    sample_dict = _build_v02_sample(i, scenario_key, rng, seed, current_time)

    # Step 1.5: Chinese-ify booking_state values FIRST (before building ScenarioFacts)
    local_rng = _random.Random(seed + i + 9999)

    # Chinese-ify current booking_state
    bs_in = sample_dict["input"]["current_state"]["booking_state"]
    city, district = _pick_city_district(local_rng)
    if bs_in.get("address") in [None, "Beijing", "Shanghai", "Hangzhou", "Guangzhou"]:
        bs_in["address"] = f"{city}{district}"
    if bs_in.get("cuisine") in [None, "Sichuan", "Cantonese", "Hunan", "Shandong"]:
        bs_in["cuisine"] = local_rng.choice(CUISINES)
    if bs_in.get("chef_name") and ("Chef" in str(bs_in.get("chef_name", "")) or not any('\u4e00' <= c <= '\u9fff' for c in str(bs_in.get("chef_name", "")))):
        bs_in["chef_name"] = _pick_chef_display(local_rng)

    # Chinese-ify expected booking_state
    bs_exp = sample_dict["expected"].get("booking_state")
    if bs_exp:
        if bs_exp.get("address") in [None, "Beijing", "Shanghai", "Hangzhou", "Guangzhou"]:
            bs_exp["address"] = f"{city}{district}"
        if bs_exp.get("cuisine") in [None, "Sichuan", "Cantonese", "Hunan", "Shandong"]:
            bs_exp["cuisine"] = bs_in.get("cuisine", local_rng.choice(CUISINES))
        if bs_exp.get("chef_name") and ("Chef" in str(bs_exp.get("chef_name", "")) or not any('\u4e00' <= c <= '\u9fff' for c in str(bs_exp.get("chef_name", "")))):
            bs_exp["chef_name"] = bs_in.get("chef_name", _pick_chef_display(local_rng))

    # Step 2: Reconstruct ScenarioFacts from the (now Chinese) sample dict
    bs_dict = sample_dict["input"]["current_state"]["booking_state"]
    bs = BookingSlot.model_validate(bs_dict)

    # Step 2.5: Recompute missing_info/info_complete from Chinese-ified booking_state
    from homechef_booking.schemas.booking import missing_required_slots
    if sample_dict["expected"].get("action") == "final":
        canonical_missing = missing_required_slots(bs)
        sample_dict["expected"]["missing_info"] = list(canonical_missing)
        sample_dict["expected"]["info_complete"] = len(canonical_missing) == 0

    # v0.2.1: Update tool_call arguments in history to match Chinese-ified booking_state
    for _msg in sample_dict["input"]["history"]:
        if _msg.get("role") == "assistant" and _msg.get("tool_calls"):
            for _tc in _msg["tool_calls"]:
                try:
                    _args = json.loads(_tc["function"]["arguments"])
                    if _args.get("address") in [None, "Beijing", "Shanghai", "Hangzhou", "Guangzhou"]:
                        _args["address"] = bs_in.get("address")
                    if _args.get("cuisine") in [None, "Sichuan", "Cantonese", "Hunan", "Shandong"]:
                        _args["cuisine"] = bs_in.get("cuisine")
                    if _args.get("chef_name") and ("Chef" in str(_args.get("chef_name", ""))):
                        _args["chef_name"] = bs_in.get("chef_name")
                    _tc["function"]["arguments"] = json.dumps(_args, ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    pass

    # Determine tool mode and status from history
    history = sample_dict["input"]["history"]
    tool_mode = ""
    tool_result_status = ""
    tool_result_payload = None
    requested_chef = None
    user_confirms = (scenario_key == "explicit_confirmation")
    user_rejects = (scenario_key == "rejection")

    for msg in history:
        role = msg.get("role", "")
        if role == "tool":
            try:
                tr = json.loads(msg["content"])
                tool_mode = tr.get("mode", "")
                tool_result_status = tr.get("status", "")
                tool_result_payload = tr
            except (json.JSONDecodeError, TypeError):
                pass
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                    if args.get("chef_name"):
                        requested_chef = args["chef_name"]
                except (json.JSONDecodeError, TypeError):
                    pass

    # Determine scenario for Chinese realization
    if scenario_key == "dietary_modification":
        realizer_scenario = "dietary_modification"
    elif scenario_key == "valid_search_tool_call" and tool_mode == "specific":
        realizer_scenario = "tool_result_specific_available"
    else:
        realizer_scenario = scenario_key

    facts = ScenarioFacts(
        booking_state=bs,
        tool_mode=tool_mode,
        tool_result_status=tool_result_status,
        tool_result_payload=tool_result_payload,
        requested_chef_name=requested_chef,
        user_confirms=user_confirms,
        user_rejects=user_rejects,
        scenario=realizer_scenario,
    )

    # Step 3: Chinese surface replacement

    # 3a: Replace user_input (allowlist affirmative for confirmation, varied rejection)
    if scenario_key == "explicit_confirmation":
        _AFFIRM = ["确认", "可以", "好的", "就这样", "确认预约"]
        sample_dict["input"]["user_input"] = _AFFIRM[(i - 1) % len(_AFFIRM)]
    elif scenario_key == "rejection":
        _REJECT = ["算了", "先不订了", "取消吧", "不用了", "还是不订了"]
        sample_dict["input"]["user_input"] = _REJECT[(i - 1) % len(_REJECT)]
    else:
        sample_dict["input"]["user_input"] = realize_chinese_user_input(
            scenario_key, facts, local_rng,
        )

    # 3b: Replace history user messages with Chinese
    new_history = []
    for msg in sample_dict["input"]["history"]:
        new_msg = dict(msg)
        if msg.get("role") == "user":
            new_msg["content"] = realize_chinese_user_input(
                scenario_key, facts, local_rng,
            )
        new_history.append(new_msg)
    sample_dict["input"]["history"] = new_history

    # 3c: Replace expected.reply with Chinese
    if sample_dict["expected"].get("action") == "final":
        reply_type = sample_dict["expected"].get("reply_type", "")
        sample_dict["expected"]["reply"] = chinese_reply(reply_type)

    # 3d: Chinese-ify booking_state display names (dietary, occasion)
    bs_in = sample_dict["input"]["current_state"]["booking_state"]
    bs_exp = sample_dict["expected"].get("booking_state", {})

    # Replace dietary_constraints (create singles AND combos for diversity)
    if bs_in.get("dietary_constraints") == ["peanut_allergy"]:
        if local_rng.random() < 0.4:
            bs_in["dietary_constraints"] = [local_rng.choice(DIETARY_SINGLE)]
        else:
            bs_in["dietary_constraints"] = list(local_rng.choice(DIETARY_COMBOS))
    if bs_in.get("dietary_constraints") == ["peanut_allergy", "halal"]:
        bs_in["dietary_constraints"] = list(local_rng.choice(DIETARY_COMBOS))
    if bs_exp.get("dietary_constraints"):
        bs_exp["dietary_constraints"] = bs_in.get("dietary_constraints", [])

    # Replace occasion (consistent between current and expected)
    if bs_in.get("occasion") in [None, "birthday"]:
        new_occasion = local_rng.choice(OCCASIONS)
        bs_in["occasion"] = new_occasion
        if bs_exp is not None:
            bs_exp["occasion"] = new_occasion

    # v0.2.1: Assign menu combinations for menu diversity (always non-empty)
    if not bs_in.get("menu"):
        bs_in["menu"] = list(local_rng.choice(MENU_COMBOS[1:]))  # skip empty combo
    if bs_exp is not None and not bs_exp.get("menu"):
        bs_exp["menu"] = bs_in.get("menu", [])

    # 3e: Replace tool_result candidate chef_names with Chinese
    for msg in sample_dict["input"]["history"]:
        if msg.get("role") == "tool":
            try:
                tr = json.loads(msg["content"])
                for field in ["candidates", "alternatives"]:
                    if field in tr:
                        for c in tr[field]:
                            if "Chef" in str(c.get("chef_name", "")):
                                c["chef_name"] = _pick_chef_display(local_rng)
                if "chef" in tr and isinstance(tr["chef"], dict):
                    if "Chef" in str(tr["chef"].get("chef_name", "")):
                        tr["chef"]["chef_name"] = _pick_chef_display(local_rng)
                msg["content"] = json.dumps(tr, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pass

    # Step 4: Add difficulty
    sample_dict["generation"]["difficulty"] = difficulty

    return sample_dict


def generate_full_raw_v02_1(output_path: Path, count: int = 600, seed: int = 3001) -> Path:
    """Generate v0.2.1 full raw data (600 rows) with Chinese + difficulty + diversity.

    Reuses the v0.2 Scenario Lineage Builder for deterministic business facts,
    then applies Chinese surface realization and difficulty assignment.
    """
    import random as _random

    rng = _random.Random(seed)
    current_time = "2026-08-12 18:00"

    # Shuffle difficulty assignments
    diff_plan = list(_V021_DIFFICULTY_PLAN[:count])
    rng.shuffle(diff_plan)

    # Scenario plan from v0.2
    plan = list(_V02_FULL_PLAN[:count])

    # Relative-time expression type cycle for search_tool_call samples
    from homechef_booking.data.chinese_realizer import (
        RELATIVE_TIME_CHINESE,
        RELATIVE_TIME_RESOLVER,
    )
    rel_types = list(RELATIVE_TIME_RESOLVER.keys())
    rel_idx = 0

    lines = []
    for i in range(1, count + 1):
        scenario_key = plan[i - 1]
        difficulty = diff_plan[i - 1]
        sample_dict = _build_v02_1_sample(i, scenario_key, difficulty, rng, seed, current_time)

        # For search_tool_call / relative_time samples, expand relative-time types
        if scenario_key == "valid_search_tool_call":
            rel_type = rel_types[rel_idx % len(rel_types)]
            rel_idx += 1
            resolved_date = RELATIVE_TIME_RESOLVER[rel_type]
            # Override service_date in current & expected booking_state
            for bs_target in [
                sample_dict["input"]["current_state"]["booking_state"],
                sample_dict["expected"].get("booking_state"),
            ]:
                if bs_target:
                    bs_target["service_date"] = resolved_date
            # Override service_date in expected.arguments (tool_call decision)
            exp_args = sample_dict["expected"].get("arguments")
            if isinstance(exp_args, dict):
                exp_args["service_date"] = resolved_date
            # Override service_date in history assistant tool_call arguments
            for hmsg in sample_dict["input"]["history"]:
                if hmsg.get("role") == "assistant" and hmsg.get("tool_calls"):
                    for htc in hmsg["tool_calls"]:
                        try:
                            hargs = json.loads(htc["function"]["arguments"])
                            hargs["service_date"] = resolved_date
                            htc["function"]["arguments"] = json.dumps(hargs, ensure_ascii=False)
                        except (json.JSONDecodeError, TypeError):
                            pass
            # Override relative_time_metadata
            from homechef_booking.data.raw_sample import RelativeTimeMetadata
            sample_dict["generation"]["relative_time_metadata"] = RelativeTimeMetadata(
                expression_type=rel_type,
                base_datetime=current_time,
                resolved_service_date=resolved_date,
                relative_expression=RELATIVE_TIME_CHINESE.get(rel_type, rel_type),
            ).model_dump()

        lines.append(json.dumps(sample_dict, ensure_ascii=False, sort_keys=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
