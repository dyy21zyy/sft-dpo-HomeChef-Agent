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
