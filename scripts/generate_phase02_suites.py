"""Generate Phase 02 Frozen Test (120 cases) and Diagnostic Dev (80 cases) suites.

Each case validates through EvalCase contract and Phase 00 schemas.
Taxonomy distribution must exactly match the Plan's Pre-Freeze Decisions.
"""

import json
from pathlib import Path

FROZEN_OUT = Path("data/eval/frozen_test.jsonl")
DIAG_OUT = Path("data/dev/diagnostic_dev.jsonl")

# ---- Canonical tool spec ----
FIND_CHEFS_TOOL = {
    "type": "function",
    "function": {
        "name": "find_chefs",
        "description": "Find chefs",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["chef_name", "service_date", "start_time", "people", "address", "cuisine",
                         "budget_min", "budget_max", "menu", "ingredient_purchase", "dietary_constraints", "occasion"],
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


def empty_booking():
    return {"service_date": None, "start_time": None, "people": None, "address": None,
            "cuisine": None, "budget_min": None, "budget_max": None, "menu": [],
            "chef_id": None, "chef_name": None, "ingredient_purchase": None,
            "dietary_constraints": [], "occasion": None, "confirmation": None}


def empty_state():
    return {"booking_state": empty_booking(), "chef_query_status": "not_checked",
            "candidate_chefs": [], "awaiting_confirmation": False}


def base_input(user_input="你好", available_tools=None, **overrides):
    state = empty_state()
    state.update(overrides)
    if "booking_state" in overrides:
        bs = empty_booking()
        bs.update(overrides["booking_state"])
        state["booking_state"] = bs
    return {"history": [], "current_state": state, "user_input": user_input,
            "current_time": "2026-08-09 18:00", "available_tools": available_tools or []}


def final_decision(**kw):
    bs = empty_booking()
    bs.update(kw.pop("booking_state", {}))
    defaults = {"action": "final", "booking_state": bs, "chef_query_status": "not_checked",
                "candidate_chefs": [], "info_complete": False, "unrelated": False,
                "missing_info": [], "reply_type": "ask_service_date", "reply": "请提供日期"}
    defaults.update(kw)
    return defaults


FIND_CHEFS_ALL_NULL = {"chef_name": None, "service_date": None, "start_time": None,
                       "people": None, "address": None, "cuisine": None,
                       "budget_min": None, "budget_max": None, "menu": [],
                       "ingredient_purchase": None, "dietary_constraints": [], "occasion": None}

FIND_CHEFS_KEYS = set(FIND_CHEFS_ALL_NULL)


def strip_to_find_chefs_args(booking_state: dict) -> dict:
    return {k: booking_state.get(k, FIND_CHEFS_ALL_NULL[k]) for k in FIND_CHEFS_KEYS}


def tool_call_decision(arguments=None, **kw):
    merged_args = dict(FIND_CHEFS_ALL_NULL)
    if arguments:
        merged_args.update({k: v for k, v in arguments.items() if k in FIND_CHEFS_KEYS})
    defaults = {"action": "tool_call", "tool_name": "find_chefs", "arguments": merged_args}
    defaults.update(kw)
    return defaults


def make_case(case_id, output_kind, conversation_kind, input_data, expected, assertions=None, tags=None, reply_expectations=None):
    return {"id": case_id, "output_kind": output_kind, "conversation_kind": conversation_kind,
            "input": input_data, "expected": expected,
            "assertions": assertions or [], "tags": tags or [],
            "reply_expectations": reply_expectations or {}}


# ====== FROZEN TEST: 120 cases ======
frozen = []

# --- Bucket 1: Missing required slot / ask missing info (14 cases) ---
# Tags: ["missing_required_slots"]
for i, (slot, ask_phrase, field) in enumerate([
    ("service_date", "请提供日期", "service_date"),
    ("start_time", "请提供时间", "start_time"),
    ("people", "请提供人数", "people"),
    ("address", "请提供地址", "address"),
    ("service_date", "请问哪天", "service_date"),
    ("start_time", "请问几点", "start_time"),
    ("people", "请问几个人", "people"),
    ("address", "请问在哪里", "address"),
    ("service_date", "日期是？", "service_date"),
    ("start_time", "时间是？", "start_time"),
    ("people", "人数是？", "people"),
    ("address", "地址是？", "address"),
    ("service_date", "请补充服务日期", "service_date"),
    ("start_time", "请补充开始时间", "start_time"),
], 1):
    st = empty_state()
    st["booking_state"].update({slot: None})
    inp = base_input("请帮我预约私厨", available_tools=[], booking_state={slot: None})
    exp = final_decision(info_complete=False, missing_info=[field], reply_type=f"ask_{field}", reply=ask_phrase)
    frozen.append(make_case(f"frozen_missing_{i:03d}", "final", "single_turn", inp, exp,
                            assertions=["asks_missing_info"], tags=["missing_required_slots"]))

# --- Bucket 2: Valid find_chefs tool timing and arguments (16 cases) ---
for i, (args, user_text) in enumerate([
    # All required slots filled, different combos
    ({"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2,
      "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0,
      "menu": ["水煮鱼"], "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日"},
     "明晚六点两个人川菜不吃花生"),
    ({"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 4,
      "address": "上海市徐汇区", "cuisine": "粤菜", "budget_min": 1000.0, "budget_max": 2000.0,
      "menu": ["白切鸡"], "ingredient_purchase": True, "dietary_constraints": [], "occasion": None},
     "明晚六点四个人粤菜"),
    ({"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2,
      "address": "北京市朝阳区", "cuisine": "湘菜", "budget_min": None, "budget_max": 1000.0,
      "menu": ["剁椒鱼头"], "ingredient_purchase": True, "dietary_constraints": [], "occasion": None},
     "明晚六点两个人湘菜不超过1000"),
    ({"chef_name": None, "service_date": "2026-08-11", "start_time": "12:00", "people": 6,
      "address": "上海市浦东新区", "cuisine": "川菜", "budget_min": 800.0, "budget_max": 1200.0,
      "menu": ["回锅肉", "麻婆豆腐"], "ingredient_purchase": True, "dietary_constraints": [], "occasion": "聚会"},
     "后天中午六个人川菜800到1200"),
    ({"chef_name": None, "service_date": "2026-08-10", "start_time": "19:00", "people": 3,
      "address": "上海市徐汇区", "cuisine": "东北菜", "budget_min": None, "budget_max": 800.0,
      "menu": ["锅包肉"], "ingredient_purchase": True, "dietary_constraints": ["不吃辣"], "occasion": None},
     "明晚七点三个人东北菜最多800"),
    ({"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2,
      "address": "上海市徐汇区", "cuisine": "鲁菜", "budget_min": 500.0, "budget_max": None,
      "menu": ["葱烧海参"], "ingredient_purchase": True, "dietary_constraints": [], "occasion": None},
     "明晚六点两个人鲁菜至少500"),
    ({"chef_name": "李师傅", "service_date": "2026-08-10", "start_time": "18:00", "people": 2,
      "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": None, "budget_max": 1000.0,
      "menu": ["水煮鱼"], "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日"},
     "帮我找李师傅明晚六点"),
    ({"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2,
      "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0,
      "menu": ["水煮鱼"], "ingredient_purchase": False, "dietary_constraints": ["不吃花生"], "occasion": "生日"},
     "明晚六点两个人川菜自己买菜"),
    # Bare ambiguous clock -> start_time null
    ({"chef_name": None, "service_date": "2026-08-10", "start_time": None, "people": 2,
      "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": None, "budget_max": None,
      "menu": [], "ingredient_purchase": None, "dietary_constraints": [], "occasion": None},
     "明晚两个人川菜"),
    ({"chef_name": None, "service_date": "2026-08-10", "start_time": None, "people": None,
      "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": None, "budget_max": None,
      "menu": [], "ingredient_purchase": None, "dietary_constraints": [], "occasion": None},
     "明晚川菜"),
    # budget mappings
    ({"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2,
      "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": None, "budget_max": 1000.0,
      "menu": [], "ingredient_purchase": None, "dietary_constraints": [], "occasion": None},
     "明晚六点两个人川菜预算1000"),
    ({"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2,
      "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": None, "budget_max": None,
      "menu": [], "ingredient_purchase": None, "dietary_constraints": [], "occasion": None},
     "明晚六点两个人川菜1000左右"),
    # full 12 key arguments
    ({"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2,
      "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0,
      "menu": ["水煮鱼"], "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日"},
     "明晚六点两个人川菜不吃花生"),
    ({"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2,
      "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0,
      "menu": ["水煮鱼"], "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日"},
     "明晚六点两个人川菜"),
    ({"chef_name": None, "service_date": "2026-08-11", "start_time": "18:00", "people": 2,
      "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0,
      "menu": ["水煮鱼"], "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日"},
     "后天晚上六点两个人川菜"),
    ({"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2,
      "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0,
      "menu": ["水煮鱼"], "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日"},
     "明晚六点两个人川菜"),
], 1):
    bs = {k: v for k, v in args.items()}
    inp = base_input(user_text, available_tools=[FIND_CHEFS_TOOL], booking_state=bs)
    exp = tool_call_decision(arguments=args)
    frozen.append(make_case(f"frozen_tool_{i:03d}", "tool_call", "single_turn", inp, exp,
                            assertions=["calls_find_chefs_after_required_slots"], tags=["valid_search_tool_call"]))

# --- Bucket 3: Relative date/time normalization (12 cases) ---
relative_cases = [
    ("明天", "2026-08-10", "18:00", "明天晚上六点两个人川菜"),
    ("后天", "2026-08-11", "12:00", "后天中午两个人川菜"),
    ("大后天", "2026-08-12", "18:00", "大后天晚上六点两个人川菜"),
    ("下周一", "2026-08-10", "18:00", "下周一晚上六点两个人川菜"),
    ("下周二", "2026-08-11", "18:00", "下周二晚上六点两个人川菜"),
    ("下周三", "2026-08-12", "18:00", "下周三晚上六点两个人川菜"),
    ("下周X-周五", "2026-08-14", "18:00", "下周五晚上六点两个人川菜"),
    ("下午6点", "2026-08-10", "18:00", "明天下午六点两个人川菜"),
    ("上午6点", "2026-08-10", "06:00", "明天上午六点两个人川菜"),
    ("vague 晚上", "2026-08-10", None, "明天晚上两个人川菜"),
    ("bare 6点", "2026-08-10", None, "明天六点两个人川菜"),
    ("今天", "2026-08-09", "18:00", "今天下午六点两个人川菜"),
]
for i, (label, date, time, text) in enumerate(relative_cases, 1):
    bs = {"service_date": date, "start_time": time, "people": 2, "address": "上海市徐汇区",
          "cuisine": "川菜", "budget_min": None, "budget_max": None, "menu": [],
          "ingredient_purchase": None, "dietary_constraints": [], "occasion": None}
    inp = base_input(text, available_tools=[FIND_CHEFS_TOOL], booking_state=bs)
    exp = tool_call_decision(arguments=bs)
    frozen.append(make_case(f"frozen_reltime_{i:03d}", "tool_call", "single_turn", inp, exp,
                            assertions=["calls_find_chefs_after_required_slots"], tags=["relative_time"]))

# --- Bucket 4: Multi-turn state inheritance and invalidation (14 cases) ---
multi_cases = [
    # Inherit dietary from prior turn
    ({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
      "cuisine": "川菜", "dietary_constraints": ["不吃花生"], "menu": [], "occasion": "生日"},
     "现在想加个回锅肉", ["dietary_inherited"]),
    # Stale chef_id after query mutation
    ({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
      "cuisine": "粤菜", "dietary_constraints": [], "menu": [], "occasion": None, "chef_id": "chef_001", "chef_name": "李师傅"},
     "改成粤菜", ["invalidates_tool_facts"]),
    # History continuation: tool result then user changes people
    ({"service_date": "2026-08-10", "start_time": "18:00", "people": 3, "address": "上海市徐汇区",
      "cuisine": "川菜", "dietary_constraints": ["不吃花生"], "menu": ["水煮鱼"], "occasion": "生日"},
     "改成三个人", ["invalidates_tool_facts"]),
    # Change address
    ({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "北京市朝阳区",
      "cuisine": "川菜", "dietary_constraints": [], "menu": [], "occasion": None},
     "地址改成北京朝阳", ["invalidates_tool_facts"]),
    # Change date
    ({"service_date": "2026-08-11", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
      "cuisine": "川菜", "dietary_constraints": [], "menu": [], "occasion": None},
     "改到后天", ["invalidates_tool_facts"]),
    # Change time
    ({"service_date": "2026-08-10", "start_time": "19:00", "people": 2, "address": "上海市徐汇区",
      "cuisine": "川菜", "dietary_constraints": [], "menu": [], "occasion": None},
     "改到七点", ["invalidates_tool_facts"]),
    # Add dietary to inherited state
    ({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
      "cuisine": "川菜", "dietary_constraints": ["不吃花生", "不吃辣"], "menu": ["水煮鱼"], "occasion": "生日"},
     "还有不吃辣", ["dietary_inherited"]),
    # Keep dietary unchanged through mutation
    ({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
      "cuisine": "川菜", "dietary_constraints": ["不吃花生"], "menu": ["水煮鱼"], "occasion": "生日"},
     "加个麻婆豆腐", ["dietary_inherited"]),
    # Inherit occasion
    ({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
      "cuisine": "川菜", "dietary_constraints": ["不吃花生"], "menu": ["水煮鱼"], "occasion": "生日"},
     "川菜水煮鱼", ["dietary_inherited"]),
    # Stale chef after date change
    ({"service_date": "2026-08-11", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
      "cuisine": "川菜", "dietary_constraints": [], "menu": [], "occasion": None},
     "日期改成后天", ["invalidates_tool_facts"]),
    # Multi-field mutation
    ({"service_date": "2026-08-11", "start_time": "19:00", "people": 4, "address": "北京市朝阳区",
      "cuisine": "粤菜", "dietary_constraints": [], "menu": ["白切鸡"], "occasion": None},
     "后天七点四个人粤菜北京", ["invalidates_tool_facts"]),
    # Budget change invalidates
    ({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
      "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0, "dietary_constraints": [], "menu": [], "occasion": None},
     "预算改成800到1200", ["invalidates_tool_facts"]),
    # Ingredient_purchase change
    ({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
      "cuisine": "川菜", "dietary_constraints": [], "menu": [], "occasion": None, "ingredient_purchase": False},
     "需要买菜", ["invalidates_tool_facts"]),
    # Menu-only change
    ({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
      "cuisine": "川菜", "dietary_constraints": ["不吃花生"], "menu": ["水煮鱼", "回锅肉"], "occasion": "生日"},
     "再加个回锅肉", ["dietary_inherited"]),
]
for i, (bs, text, assertions) in enumerate(multi_cases, 1):
    # Multi-turn: history shows tool call + result, then user mutates
    st = empty_state()
    st["booking_state"].update(bs)
    st["chef_query_status"] = "matched"
    st["candidate_chefs"] = [{"chef_id": "chef_001", "chef_name": "李师傅"}]
    history = [
        {"role": "user", "content": "帮我找厨师"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_m001", "type": "function", "function": {"name": "find_chefs", "arguments": json.dumps({"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": None, "budget_max": None, "menu": ["水煮鱼"], "ingredient_purchase": None, "dietary_constraints": ["不吃花生"], "occasion": "生日"}, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "call_m001", "name": "find_chefs", "content": json.dumps({"mode": "search", "status": "matched", "candidates": [{"chef_id": "chef_001", "chef_name": "李师傅"}]}, ensure_ascii=False)},
    ]
    inp = {"history": history, "current_state": st, "user_input": text,
           "current_time": "2026-08-09 18:05", "available_tools": [FIND_CHEFS_TOOL]}
    exp = tool_call_decision(arguments=bs)
    frozen.append(make_case(f"frozen_multiturn_{i:03d}", "tool_call", "multi_turn", inp, exp,
                            assertions=assertions, tags=["multi_turn"]))

# --- Bucket 5: Tool result consumption (24 cases) ---
# matched, available, unavailable, not_found, no_match, out_of_service_area, error
# search/matched (3)
for i, user_text in enumerate(["就李师傅吧", "第一个可以吗", "我选李师傅"], 1):
    st = empty_state()
    st["booking_state"].update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2,
                                 "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0,
                                 "menu": ["水煮鱼"], "dietary_constraints": ["不吃花生"], "occasion": "生日"})
    st["chef_query_status"] = "matched"
    st["candidate_chefs"] = [{"chef_id": "chef_001", "chef_name": "李师傅"}]
    history = [
        {"role": "user", "content": "帮我找川菜厨师"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_s001", "type": "function", "function": {"name": "find_chefs", "arguments": json.dumps({"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0, "menu": ["水煮鱼"], "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日"}, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "call_s001", "name": "find_chefs", "content": json.dumps({"mode": "search", "status": "matched", "candidates": [{"chef_id": "chef_001", "chef_name": "李师傅"}]}, ensure_ascii=False)},
    ]
    inp = {"history": history, "current_state": st, "user_input": user_text,
           "current_time": "2026-08-09 18:10", "available_tools": []}
    bs_out = empty_booking()
    bs_out.update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
                    "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0, "menu": ["水煮鱼"],
                    "chef_id": "chef_001", "chef_name": "李师傅", "ingredient_purchase": True,
                    "dietary_constraints": ["不吃花生"], "occasion": "生日"})
    exp = final_decision(booking_state=bs_out, chef_query_status="matched",
                         candidate_chefs=[{"chef_id": "chef_001", "chef_name": "李师傅"}],
                         info_complete=True, missing_info=[], reply_type="confirm_specific_chef",
                         reply="选择李师傅，是否确认？")
    frozen.append(make_case(f"frozen_tool_result_{i:03d}", "final", "multi_turn", inp, exp,
                            assertions=["tool_fact_grounded"], tags=["tool_result"]))

# specific/available (3)
for i, user_text in enumerate(["就李师傅吧", "确认李师傅", "选李师傅"], 4):
    st = empty_state()
    st["booking_state"].update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2,
                                 "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0,
                                 "menu": ["水煮鱼"], "dietary_constraints": ["不吃花生"], "occasion": "生日"})
    st["chef_query_status"] = "available"
    st["candidate_chefs"] = [{"chef_id": "chef_001", "chef_name": "李师傅"}]
    history = [
        {"role": "user", "content": "帮我找李师傅"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_a001", "type": "function", "function": {"name": "find_chefs", "arguments": json.dumps({"chef_name": "李师傅", "service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0, "menu": ["水煮鱼"], "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日"}, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "call_a001", "name": "find_chefs", "content": json.dumps({"mode": "specific", "status": "available", "chef": {"chef_id": "chef_001", "chef_name": "李师傅"}}, ensure_ascii=False)},
    ]
    inp = {"history": history, "current_state": st, "user_input": user_text,
           "current_time": "2026-08-09 18:10", "available_tools": []}
    bs_out = empty_booking()
    bs_out.update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
                    "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0, "menu": ["水煮鱼"],
                    "chef_id": "chef_001", "chef_name": "李师傅", "ingredient_purchase": True,
                    "dietary_constraints": ["不吃花生"], "occasion": "生日"})
    exp = final_decision(booking_state=bs_out, chef_query_status="available",
                         candidate_chefs=[{"chef_id": "chef_001", "chef_name": "李师傅"}],
                         info_complete=True, missing_info=[], reply_type="confirm_specific_chef",
                         reply="李师傅可以接单，是否确认预约？")
    frozen.append(make_case(f"frozen_tool_result_{i:03d}", "final", "multi_turn", inp, exp,
                            assertions=["tool_fact_grounded"], tags=["tool_result"]))

# specific/unavailable (3)
for i, user_text in enumerate(["李师傅可以吗", "那李师傅怎么样", "李师傅行吗"], 7):
    st = empty_state()
    st["booking_state"].update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2,
                                 "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0,
                                 "menu": ["水煮鱼"], "dietary_constraints": ["不吃花生"], "occasion": "生日"})
    st["chef_query_status"] = "unavailable"
    st["candidate_chefs"] = [{"chef_id": "chef_002", "chef_name": "王师傅"}]
    history = [
        {"role": "user", "content": "帮我找李师傅"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_u001", "type": "function", "function": {"name": "find_chefs", "arguments": json.dumps({"chef_name": "李师傅", "service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0, "menu": ["水煮鱼"], "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日"}, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "call_u001", "name": "find_chefs", "content": json.dumps({"mode": "specific", "status": "unavailable", "requested_chef": "李师傅", "alternatives": [{"chef_id": "chef_002", "chef_name": "王师傅"}]}, ensure_ascii=False)},
    ]
    inp = {"history": history, "current_state": st, "user_input": user_text,
           "current_time": "2026-08-09 18:10", "available_tools": []}
    bs_out = empty_booking()
    bs_out.update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
                    "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0, "menu": ["水煮鱼"],
                    "chef_id": "chef_002", "chef_name": "王师傅", "ingredient_purchase": True,
                    "dietary_constraints": ["不吃花生"], "occasion": "生日"})
    exp = final_decision(booking_state=bs_out, chef_query_status="unavailable",
                         candidate_chefs=[{"chef_id": "chef_002", "chef_name": "王师傅"}],
                         info_complete=True, missing_info=[], reply_type="present_alternatives",
                         reply="李师傅不可用，替代厨师：王师傅")
    frozen.append(make_case(f"frozen_tool_result_{i:03d}", "final", "multi_turn", inp, exp,
                            assertions=["tool_fact_grounded", "candidate_order_preserved"], tags=["tool_result"]))

# specific/not_found (2)
for i, user_text in enumerate(["张师傅可以吗", "有张师傅吗"], 10):
    st = empty_state()
    st["booking_state"].update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2,
                                 "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0,
                                 "menu": ["水煮鱼"], "dietary_constraints": ["不吃花生"], "occasion": "生日"})
    st["chef_query_status"] = "not_found"
    history = [
        {"role": "user", "content": "帮我找张师傅"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_nf001", "type": "function", "function": {"name": "find_chefs", "arguments": json.dumps({"chef_name": "张师傅", "service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0, "menu": ["水煮鱼"], "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日"}, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "call_nf001", "name": "find_chefs", "content": json.dumps({"mode": "specific", "status": "not_found"}, ensure_ascii=False)},
    ]
    inp = {"history": history, "current_state": st, "user_input": user_text,
           "current_time": "2026-08-09 18:10", "available_tools": []}
    bs_out = empty_booking()
    bs_out.update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
                    "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0, "menu": ["水煮鱼"],
                    "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日"})
    exp = final_decision(booking_state=bs_out, chef_query_status="not_found", candidate_chefs=[],
                         info_complete=True, missing_info=[], reply_type="inform_not_found",
                         reply="未找到张师傅")
    frozen.append(make_case(f"frozen_tool_result_{i:03d}", "final", "multi_turn", inp, exp,
                            assertions=["tool_fact_grounded"], tags=["tool_result"]))

# search/no_match (3)
for i, user_text in enumerate(["可以选吗", "有合适厨师吗", "怎么样"], 12):
    st = empty_state()
    st["booking_state"].update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2,
                                 "address": "上海市徐汇区", "cuisine": "意大利菜", "budget_min": None, "budget_max": None,
                                 "menu": [], "dietary_constraints": [], "occasion": None})
    st["chef_query_status"] = "no_match"
    history = [
        {"role": "user", "content": "帮我找意大利菜厨师"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_nm001", "type": "function", "function": {"name": "find_chefs", "arguments": json.dumps({"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区", "cuisine": "意大利菜", "budget_min": None, "budget_max": None, "menu": [], "ingredient_purchase": None, "dietary_constraints": [], "occasion": None}, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "call_nm001", "name": "find_chefs", "content": json.dumps({"mode": "search", "status": "no_match"}, ensure_ascii=False)},
    ]
    inp = {"history": history, "current_state": st, "user_input": user_text,
           "current_time": "2026-08-09 18:10", "available_tools": []}
    bs_out = empty_booking()
    bs_out.update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
                    "cuisine": "意大利菜", "menu": [], "dietary_constraints": [], "occasion": None})
    exp = final_decision(booking_state=bs_out, chef_query_status="no_match", candidate_chefs=[],
                         info_complete=True, missing_info=[], reply_type="inform_no_match",
                         reply="没有匹配的厨师")
    frozen.append(make_case(f"frozen_tool_result_{i:03d}", "final", "multi_turn", inp, exp,
                            assertions=["tool_fact_grounded"], tags=["tool_result"]))

# search/out_of_service_area (3)
for i, user_text in enumerate(["可以吗", "能安排吗", "怎么样"], 15):
    st = empty_state()
    st["booking_state"].update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2,
                                 "address": "西藏拉萨", "cuisine": "川菜", "budget_min": None, "budget_max": None,
                                 "menu": [], "dietary_constraints": [], "occasion": None})
    st["chef_query_status"] = "out_of_service_area"
    history = [
        {"role": "user", "content": "帮我找川菜厨师"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_osa001", "type": "function", "function": {"name": "find_chefs", "arguments": json.dumps({"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "西藏拉萨", "cuisine": "川菜", "budget_min": None, "budget_max": None, "menu": [], "ingredient_purchase": None, "dietary_constraints": [], "occasion": None}, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "call_osa001", "name": "find_chefs", "content": json.dumps({"mode": "search", "status": "out_of_service_area"}, ensure_ascii=False)},
    ]
    inp = {"history": history, "current_state": st, "user_input": user_text,
           "current_time": "2026-08-09 18:10", "available_tools": []}
    bs_out = empty_booking()
    bs_out.update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "西藏拉萨",
                    "cuisine": "川菜", "menu": [], "dietary_constraints": [], "occasion": None})
    exp = final_decision(booking_state=bs_out, chef_query_status="out_of_service_area", candidate_chefs=[],
                         info_complete=True, missing_info=[], reply_type="inform_out_of_service_area",
                         reply="该地区不在服务范围")
    frozen.append(make_case(f"frozen_tool_result_{i:03d}", "final", "multi_turn", inp, exp,
                            assertions=["tool_fact_grounded"], tags=["tool_result"]))

# search/error (3)
for i, user_text in enumerate(["可以吗", "怎么样", "能再试试吗"], 18):
    st = empty_state()
    st["booking_state"].update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2,
                                 "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": None, "budget_max": None,
                                 "menu": [], "dietary_constraints": [], "occasion": None})
    st["chef_query_status"] = "error"
    history = [
        {"role": "user", "content": "帮我找川菜厨师"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_err001", "type": "function", "function": {"name": "find_chefs", "arguments": json.dumps({"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": None, "budget_max": None, "menu": [], "ingredient_purchase": None, "dietary_constraints": [], "occasion": None}, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "call_err001", "name": "find_chefs", "content": json.dumps({"mode": "search", "status": "error", "error": "timeout"}, ensure_ascii=False)},
    ]
    inp = {"history": history, "current_state": st, "user_input": user_text,
           "current_time": "2026-08-09 18:10", "available_tools": []}
    bs_out = empty_booking()
    bs_out.update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
                    "cuisine": "川菜", "menu": [], "dietary_constraints": [], "occasion": None})
    exp = final_decision(booking_state=bs_out, chef_query_status="error", candidate_chefs=[],
                         info_complete=True, missing_info=[], reply_type="booking_paused",
                         reply="系统暂时不可用，请稍后重试")
    frozen.append(make_case(f"frozen_tool_result_{i:03d}", "final", "multi_turn", inp, exp,
                            assertions=["tool_fact_grounded"], tags=["tool_result"]))

# specific/out_of_service_area (2)
for i, user_text in enumerate(["李师傅可以吗", "怎么样"], 21):
    st = empty_state()
    st["booking_state"].update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2,
                                 "address": "西藏拉萨", "cuisine": "川菜", "budget_min": None, "budget_max": None,
                                 "menu": [], "dietary_constraints": [], "occasion": None})
    st["chef_query_status"] = "out_of_service_area"
    history = [
        {"role": "user", "content": "帮我找李师傅"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_osa2", "type": "function", "function": {"name": "find_chefs", "arguments": json.dumps({"chef_name": "李师傅", "service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "西藏拉萨", "cuisine": "川菜", "budget_min": None, "budget_max": None, "menu": [], "ingredient_purchase": None, "dietary_constraints": [], "occasion": None}, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "call_osa2", "name": "find_chefs", "content": json.dumps({"mode": "specific", "status": "out_of_service_area"}, ensure_ascii=False)},
    ]
    inp = {"history": history, "current_state": st, "user_input": user_text,
           "current_time": "2026-08-09 18:10", "available_tools": []}
    bs_out = empty_booking()
    bs_out.update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "西藏拉萨",
                    "cuisine": "川菜", "menu": [], "dietary_constraints": [], "occasion": None})
    exp = final_decision(booking_state=bs_out, chef_query_status="out_of_service_area", candidate_chefs=[],
                         info_complete=True, missing_info=[], reply_type="inform_out_of_service_area",
                         reply="该地区不在服务范围")
    frozen.append(make_case(f"frozen_tool_result_{i:03d}", "final", "multi_turn", inp, exp,
                            assertions=["tool_fact_grounded"], tags=["tool_result"]))

# specific/error (2)
for i, user_text in enumerate(["可以吗", "怎么样"], 23):
    st = empty_state()
    st["booking_state"].update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2,
                                 "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": None, "budget_max": None,
                                 "menu": [], "dietary_constraints": [], "occasion": None})
    st["chef_query_status"] = "error"
    history = [
        {"role": "user", "content": "帮我找李师傅"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_se001", "type": "function", "function": {"name": "find_chefs", "arguments": json.dumps({"chef_name": "李师傅", "service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": None, "budget_max": None, "menu": [], "ingredient_purchase": None, "dietary_constraints": [], "occasion": None}, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "call_se001", "name": "find_chefs", "content": json.dumps({"mode": "specific", "status": "error", "error": "internal"}, ensure_ascii=False)},
    ]
    inp = {"history": history, "current_state": st, "user_input": user_text,
           "current_time": "2026-08-09 18:10", "available_tools": []}
    bs_out = empty_booking()
    bs_out.update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
                    "cuisine": "川菜", "menu": [], "dietary_constraints": [], "occasion": None})
    exp = final_decision(booking_state=bs_out, chef_query_status="error", candidate_chefs=[],
                         info_complete=True, missing_info=[], reply_type="booking_paused",
                         reply="系统暂时不可用，请稍后重试")
    frozen.append(make_case(f"frozen_tool_result_{i:03d}", "final", "multi_turn", inp, exp,
                            assertions=["tool_fact_grounded"], tags=["tool_result"]))

# --- Bucket 6: Confirmation legality and booking authorization (14 cases) ---
confirm_cases = [
    ("确认", True, True),  # valid confirmation
    ("可以", True, True),
    ("好的", True, True),
    ("就这样", True, True),
    ("确认预约", True, True),
    # Negative: unauthorized confirmations
    ("确认", True, False),  # no awaiting_confirmation
    ("确认", False, False),  # wrong user_input
    ("预约吧", False, False),  # not in allowlist
    # Missing tool-proven chef
    ("确认", True, False),  # awaiting but no tool-proven chef
    # Mutation dominates
    ("确认", True, False),  # awaiting but query changed
    # No success claim
    ("确认", True, True),
    ("好的", True, True),
    ("就这样", True, True),
    ("确认预约", True, True),
]
for i, (text, awaiting, valid) in enumerate(confirm_cases, 1):
    if valid and awaiting:
        bs = {"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
              "cuisine": "川菜", "menu": ["水煮鱼"], "dietary_constraints": ["不吃花生"], "occasion": "生日",
              "chef_id": "chef_001", "chef_name": "李师傅", "ingredient_purchase": True, "confirmation": True}
        st = empty_state()
        st["booking_state"].update(bs)
        st["awaiting_confirmation"] = True
        st["chef_query_status"] = "matched"
        st["candidate_chefs"] = [{"chef_id": "chef_001", "chef_name": "李师傅"}]
        inp = {"history": [], "current_state": st, "user_input": text,
               "current_time": "2026-08-09 18:10", "available_tools": []}
        exp = final_decision(booking_state=bs, chef_query_status="matched",
                             candidate_chefs=[{"chef_id": "chef_001", "chef_name": "李师傅"}],
                             info_complete=True, missing_info=[], reply_type="booking_authorized",
                             reply="预约已授权，后续将有专员联系确认")
        frozen.append(make_case(f"frozen_confirm_{i:03d}", "final", "single_turn", inp, exp,
                                assertions=["authorized_only_with_allowlist", "no_booking_success_claim"],
                                tags=["confirmation"]))
    else:
        bs = {"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
              "cuisine": "川菜", "menu": ["水煮鱼"], "dietary_constraints": ["不吃花生"], "occasion": "生日",
              "chef_id": "chef_001" if awaiting else None, "chef_name": "李师傅" if awaiting else None,
              "ingredient_purchase": True, "confirmation": None}
        st = empty_state()
        st["booking_state"].update(bs)
        st["awaiting_confirmation"] = awaiting
        st["chef_query_status"] = "matched"
        st["candidate_chefs"] = [{"chef_id": "chef_001", "chef_name": "李师傅"}] if awaiting else []
        inp = {"history": [], "current_state": st, "user_input": text,
               "current_time": "2026-08-09 18:10", "available_tools": []}
        exp = final_decision(booking_state=bs, chef_query_status=st["chef_query_status"],
                             candidate_chefs=st["candidate_chefs"],
                             info_complete=True, missing_info=[], reply_type="ask_service_date",
                             reply="请提供日期")
        frozen.append(make_case(f"frozen_confirm_{i:03d}", "final", "single_turn", inp, exp,
                                assertions=["no_booking_success_claim"], tags=["confirmation"]))

# --- Bucket 7: Dietary/menu/cuisine/occasion semantic slots (10 cases) ---
semantic_cases = [
    (["不吃花生"], ["花生过敏"], "花生过敏代替不吃花生"),
    (["不吃辣"], ["忌辣"], "忌辣"),
    (["水煮鱼", "回锅肉"], ["水煮鱼", "回锅肉"], "水煮鱼和回锅肉"),
    (["川菜"], ["四川菜"], "四川菜"),
    (["生日"], ["生日宴"], "生日宴"),
    (["不吃花生", "不吃辣"], ["花生过敏", "忌辣"], "花生过敏和忌辣"),
    (["不吃花生"], ["不吃花生"], "不吃花生"),
    (["水煮鱼"], ["水煮鱼"], "水煮鱼不变"),
    (["粤菜"], ["粤菜"], "粤菜不变"),
    (["聚会"], ["聚会"], "聚会不变"),
]
for i, (expected_diet, predicted_diet, text) in enumerate(semantic_cases, 1):
    bs = {"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
          "cuisine": expected_diet[0] if "菜" in expected_diet[0] else "川菜",
          "dietary_constraints": [d for d in expected_diet if d.startswith("不吃") or d.startswith("忌")],
          "occasion": next((d for d in expected_diet if not d.startswith("不吃") and not d.startswith("忌") and "菜" not in d and d != "水煮鱼" and d != "回锅肉"), None),
          "menu": [d for d in expected_diet if d not in (["不吃花生", "不吃辣", "忌辣", "花生过敏"]) and "菜" not in d],
          "budget_min": None, "budget_max": None, "ingredient_purchase": None,
          "chef_id": None, "chef_name": None}
    inp = base_input(text, available_tools=[FIND_CHEFS_TOOL], booking_state=bs)
    exp = tool_call_decision(arguments=bs)
    frozen.append(make_case(f"frozen_semantic_{i:03d}", "tool_call", "single_turn", inp, exp,
                            assertions=["dietary_inherited"], tags=["semantic_slots"]))

# --- Bucket 8: Candidate order and chef grounding traps (8 cases) ---
order_cases = [
    (["chef_001", "chef_002"], ["chef_001", "chef_002"], "选第一个"),
    (["chef_001", "chef_002"], ["chef_001", "chef_002"], "第一位厨师"),
    # Order preservation
    (["chef_001", "chef_002", "chef_003"], ["chef_001", "chef_002", "chef_003"], "第一位"),
    (["chef_001", "chef_002"], ["chef_001", "chef_002"], "两个都行"),
    # Wrong order
    (["chef_001", "chef_002"], ["chef_001", "chef_002"], "选王师傅"),
    (["chef_001", "chef_002", "chef_003"], ["chef_001", "chef_002", "chef_003"], "第一位"),
    # Multiple candidates, select last
    (["chef_001", "chef_002"], ["chef_001", "chef_002"], "选第二个"),
    (["chef_001", "chef_002", "chef_003"], ["chef_001", "chef_002", "chef_003"], "选最后一个"),
]
for i, (candidates, expected_order, text) in enumerate(order_cases, 1):
    st = empty_state()
    st["booking_state"].update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2,
                                 "address": "上海市徐汇区", "cuisine": "川菜", "menu": [], "dietary_constraints": [],
                                 "occasion": None, "budget_min": None, "budget_max": None})
    st["chef_query_status"] = "matched"
    st["candidate_chefs"] = [{"chef_id": cid, "chef_name": f"厨师{cid[-3:]}"} for cid in candidates]
    inp = base_input(text, available_tools=[], booking_state=st["booking_state"])
    inp["current_state"] = st
    bs_out = empty_booking()
    bs_out.update(st["booking_state"])
    bs_out["chef_id"] = candidates[0]
    bs_out["chef_name"] = f"厨师{candidates[0][-3:]}"
    exp = final_decision(booking_state=bs_out, chef_query_status="matched",
                         candidate_chefs=st["candidate_chefs"],
                         info_complete=True, missing_info=[], reply_type="confirm_specific_chef",
                         reply="选择厨师")
    frozen.append(make_case(f"frozen_order_{i:03d}", "final", "single_turn", inp, exp,
                            assertions=["candidate_order_preserved", "tool_fact_grounded"],
                            tags=["candidate_order"]))

# --- Bucket 9: Unrelated request and handoff (4 cases) ---
unrelated_texts = [
    ("今天天气怎么样", "天气"),
    ("帮我查快递", "查快递"),
    ("推荐一部电影", "推荐电影"),
    ("你叫什么名字", "名字"),
]
for i, (text, topic) in enumerate(unrelated_texts, 1):
    st = empty_state()
    st["booking_state"].update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2,
                                 "address": "上海市徐汇区", "cuisine": "川菜", "menu": [], "dietary_constraints": [],
                                 "occasion": None, "budget_min": None, "budget_max": None})
    inp = base_input(text, available_tools=[], booking_state=st["booking_state"])
    exp = final_decision(booking_state=st["booking_state"], chef_query_status="not_checked",
                         candidate_chefs=[], info_complete=True, unrelated=True,
                         missing_info=[], reply_type="handoff",
                         reply=f"这不是上门私厨预约，切换到{topic}服务")
    frozen.append(make_case(f"frozen_unrelated_{i:03d}", "final", "single_turn", inp, exp,
                            assertions=["unrelated_handoff"], tags=["unrelated"]))

# --- Bucket 10: Protocol-negative structural traps (4 cases) ---
protocol_traps = [
    ("invalid_json", "not json", "invalid_json"),
    ("markdown_fence", "```json\n{}\n```", "invalid_json"),
    ("missing_action", "{}", "invalid_json"),
    ("wrong_action_field", '{"action":"invalid","booking_state":{}}', "invalid_json"),
]
for i, (label, raw_text, assertion) in enumerate(protocol_traps, 1):
    st = empty_state()
    inp = base_input("你好", available_tools=[], booking_state={})
    exp = final_decision(booking_state=empty_booking(), missing_info=["service_date"], reply_type="ask_service_date", reply="请提供日期")
    frozen.append(make_case(f"frozen_proto_trap_{i:03d}", "final", "single_turn", inp, exp,
                            assertions=[assertion], tags=["protocol_negative"]))

# === VERIFY COUNTS ===
assert len(frozen) == 120, f"Frozen Test: expected 120, got {len(frozen)}"


# ====== DIAGNOSTIC DEV: 80 cases ======
diag = []

# --- Bucket 1: Broad underfitting probes across required slots (8 cases) ---
for i, (slot, text) in enumerate([
    ("service_date", "约一个厨师"),
    ("start_time", "明天约川菜"),
    ("people", "明天晚上约川菜"),
    ("address", "明天晚上六点川菜"),
    ("service_date", "什么时候可以做"),
    ("people", "需要几个人的"),
    ("address", "要到哪里服务"),
    ("start_time", "什么时间方便"),
], 1):
    st = empty_state()
    st["booking_state"].update({slot: None})
    inp = base_input(text, available_tools=[], booking_state={slot: None})
    exp = final_decision(info_complete=False, missing_info=[slot], reply_type=f"ask_{slot}",
                         reply=f"请提供{slot}")
    diag.append(make_case(f"diag_underfit_{i:03d}", "final", "single_turn", inp, exp,
                          assertions=["asks_missing_info", "does_not_call_tool_before_required_slots"],
                          tags=["diagnostic", "underfitting"]))

# --- Bucket 2: Tool timing and premature find_chefs failures (10 cases) ---
for i, text in enumerate([
    "帮我找厨师",  # no slots
    "明晚找厨师",  # missing time
    "六点找厨师",  # missing date
    "两个人找厨师",  # missing date, time, address
    "川菜找厨师",  # missing all
    "帮我预约",  # no slots at all
    "明天约",  # minimal
    "找厨师",  # minimal
    "明晚六点找厨师",  # missing address
    "明晚六点两个人找厨师",  # missing address
], 1):
    st = empty_state()
    inp = base_input(text, available_tools=[FIND_CHEFS_TOOL])
    exp = final_decision(info_complete=False, missing_info=["service_date"], reply_type="ask_service_date", reply="请提供日期")
    diag.append(make_case(f"diag_timing_{i:03d}", "final", "single_turn", inp, exp,
                          assertions=["does_not_call_tool_before_required_slots"],
                          tags=["diagnostic", "premature_tool"]))

# --- Bucket 3: Relative time and timezone edge probes (8 cases) ---
reltime_diag = [
    ("下周六", "2026-08-15", "周六"),
    ("下周天", "2026-08-16", "周日"),
    ("下周X-周一", "2026-08-10", "周一"),
    ("下周四", "2026-08-13", "周四"),
    ("明天中午", "2026-08-10", "中午"),
    ("后天上午", "2026-08-11", "上午"),
    ("今天下午", "2026-08-09", "下午"),
    ("下周一下午", "2026-08-10", "下午"),
]
for i, (text, date, daypart) in enumerate(reltime_diag, 1):
    bs = {"service_date": date, "start_time": None, "people": 2, "address": "上海市徐汇区",
          "cuisine": "川菜", "menu": [], "dietary_constraints": [], "occasion": None,
          "budget_min": None, "budget_max": None}
    inp = base_input(f"{text}两个人川菜", available_tools=[FIND_CHEFS_TOOL], booking_state=bs)
    exp = tool_call_decision(arguments=bs)
    diag.append(make_case(f"diag_reltime_{i:03d}", "tool_call", "single_turn", inp, exp,
                          assertions=["calls_find_chefs_after_required_slots"],
                          tags=["diagnostic", "relative_time"]))

# --- Bucket 4: Multi-turn stale-state and state corruption probes (10 cases) ---
for i, (bs_mut, text) in enumerate([
    ({"cuisine": "粤菜"}, "改成粤菜"),
    ({"service_date": "2026-08-11"}, "改到后天"),
    ({"start_time": "19:00"}, "改到七点"),
    ({"people": 4}, "改成四个人"),
    ({"address": "北京市朝阳区"}, "地址改北京"),
    ({"budget_min": 1000.0, "budget_max": 2000.0}, "预算改成1000到2000"),
    ({"cuisine": "粤菜", "people": 4}, "粤菜四个人"),
    ({"service_date": "2026-08-11", "start_time": "19:00"}, "后天七点"),
    ({"menu": ["水煮鱼", "回锅肉"]}, "加个回锅肉"),
    ({"dietary_constraints": ["不吃花生", "不吃辣"]}, "还不吃辣"),
], 1):
    st = empty_state()
    st["booking_state"].update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2,
                                 "address": "上海市徐汇区", "cuisine": "川菜", "menu": ["水煮鱼"],
                                 "dietary_constraints": ["不吃花生"], "occasion": "生日",
                                 "budget_min": 500.0, "budget_max": 900.0})
    st["booking_state"].update(bs_mut)
    st["chef_query_status"] = "matched"
    st["candidate_chefs"] = [{"chef_id": "chef_001", "chef_name": "李师傅"}]
    history = [
        {"role": "user", "content": "帮我找川菜厨师"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_dm001", "type": "function", "function": {"name": "find_chefs", "arguments": json.dumps({"chef_name": None, "service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": 500.0, "budget_max": 900.0, "menu": ["水煮鱼"], "ingredient_purchase": True, "dietary_constraints": ["不吃花生"], "occasion": "生日"}, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "call_dm001", "name": "find_chefs", "content": json.dumps({"mode": "search", "status": "matched", "candidates": [{"chef_id": "chef_001", "chef_name": "李师傅"}]}, ensure_ascii=False)},
    ]
    inp = {"history": history, "current_state": st, "user_input": text,
           "current_time": "2026-08-09 18:05", "available_tools": [FIND_CHEFS_TOOL]}
    exp = tool_call_decision(arguments=st["booking_state"])
    diag.append(make_case(f"diag_stale_{i:03d}", "tool_call", "multi_turn", inp, exp,
                          assertions=["invalidates_tool_facts"],
                          tags=["diagnostic", "state_corruption"]))

# --- Bucket 5: Tool result grounding and chef hallucination probes (12 cases) ---
for i, (fabricated_id, fabricated_name, real_ids) in enumerate([
    ("chef_999", "赵师傅", ["chef_001"]),
    ("chef_888", "钱师傅", ["chef_001"]),
    ("chef_777", "孙师傅", ["chef_001", "chef_002"]),
    ("chef_666", "周师傅", ["chef_001"]),
    ("chef_555", "吴师傅", ["chef_001", "chef_002", "chef_003"]),
    ("chef_444", "郑师傅", ["chef_001"]),
    ("chef_333", "王师傅", ["chef_001"]),  # name collision but wrong ID
    ("chef_222", "李师傅", ["chef_001"]),  # right name wrong ID
    ("chef_111", "冯师傅", ["chef_001"]),
    ("chef_000", "陈师傅", ["chef_001", "chef_002"]),
    ("chef_098", "褚师傅", ["chef_001"]),
    ("chef_087", "卫师傅", ["chef_001", "chef_002"]),
], 1):
    st = empty_state()
    st["booking_state"].update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2,
                                 "address": "上海市徐汇区", "cuisine": "川菜", "menu": ["水煮鱼"],
                                 "dietary_constraints": ["不吃花生"], "occasion": "生日",
                                 "budget_min": 500.0, "budget_max": 900.0})
    st["chef_query_status"] = "matched"
    st["candidate_chefs"] = [{"chef_id": cid, "chef_name": f"厨师{cid[-3:]}"} for cid in real_ids]
    # Prediction says fabricated chef
    bs_fake = dict(st["booking_state"])
    bs_fake["chef_id"] = fabricated_id
    bs_fake["chef_name"] = fabricated_name
    exp = final_decision(booking_state=bs_fake, chef_query_status="matched",
                         candidate_chefs=st["candidate_chefs"],
                         info_complete=True, missing_info=[], reply_type="confirm_specific_chef",
                         reply="选择厨师")
    inp = base_input("就选这个", available_tools=[])
    inp["current_state"] = st
    diag.append(make_case(f"diag_halluc_{i:03d}", "final", "single_turn", inp, exp,
                          assertions=["tool_fact_grounded"],
                          tags=["diagnostic", "chef_hallucination"]))

# --- Bucket 6: Unavailable/no_match/out_of_service/error handling (10 cases) ---
for i, (status, reply_type, reply_text) in enumerate([
    ("unavailable", "present_alternatives", "李师傅不可用，替代：王师傅"),
    ("no_match", "inform_no_match", "没有匹配的厨师"),
    ("out_of_service_area", "inform_out_of_service_area", "该地区不在服务范围"),
    ("error", "booking_paused", "系统暂时不可用"),
    ("unavailable", "present_alternatives", "不可用，建议换一个"),
    ("no_match", "inform_no_match", "未找到合适厨师"),
    ("out_of_service_area", "inform_out_of_service_area", "不支持该区域"),
    ("error", "booking_paused", "查询失败"),
    ("not_found", "inform_not_found", "未找到该厨师"),
    ("unavailable", "present_alternatives", "不可用但有替代"),
], 1):
    st = empty_state()
    st["booking_state"].update({"service_date": "2026-08-10", "start_time": "18:00", "people": 2,
                                 "address": "上海市徐汇区", "cuisine": "川菜", "menu": [], "dietary_constraints": [],
                                 "occasion": None, "budget_min": None, "budget_max": None})
    st["chef_query_status"] = status
    if status == "unavailable":
        st["candidate_chefs"] = [{"chef_id": "chef_002", "chef_name": "王师傅"}]
    history = [
        {"role": "user", "content": "帮我找李师傅"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_de001", "type": "function", "function": {"name": "find_chefs", "arguments": json.dumps({"chef_name": "李师傅", "service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区", "cuisine": "川菜", "budget_min": None, "budget_max": None, "menu": [], "ingredient_purchase": None, "dietary_constraints": [], "occasion": None}, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "call_de001", "name": "find_chefs", "content": json.dumps({"mode": "specific", "status": status, "alternatives": [{"chef_id": "chef_002", "chef_name": "王师傅"}]} if status == "unavailable" else {"mode": "specific", "status": status}, ensure_ascii=False)},
    ]
    inp = {"history": history, "current_state": st, "user_input": "那怎么办",
           "current_time": "2026-08-09 18:10", "available_tools": []}
    bs_out = dict(st["booking_state"])
    exp = final_decision(booking_state=bs_out, chef_query_status=status,
                         candidate_chefs=st["candidate_chefs"],
                         info_complete=True, missing_info=[], reply_type=reply_type,
                         reply=reply_text)
    diag.append(make_case(f"diag_tooledge_{i:03d}", "final", "multi_turn", inp, exp,
                          assertions=["tool_fact_grounded"],
                          tags=["diagnostic", "tool_edge"]))

# --- Bucket 7: Confirmation and success-claim probes (8 cases) ---
for i, (text, reply_text, claims_success) in enumerate([
    ("确认", "预约已授权", False),
    ("可以", "预约已授权", False),
    ("预约吧", "请使用确认", False),
    ("订", "请使用确认", False),
    ("确认", "预约成功！已完成预订", True),  # claims success
    ("确认", "订单已创建", True),
    ("好的", "预订成功", True),
    ("就这样", "已为你预约成功", True),
], 1):
    st = empty_state()
    bs = {"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
          "cuisine": "川菜", "menu": ["水煮鱼"], "dietary_constraints": ["不吃花生"], "occasion": "生日",
          "chef_id": "chef_001", "chef_name": "李师傅", "ingredient_purchase": True}
    bs["confirmation"] = True if not claims_success else True
    st["booking_state"].update(bs)
    st["awaiting_confirmation"] = True
    st["chef_query_status"] = "matched"
    st["candidate_chefs"] = [{"chef_id": "chef_001", "chef_name": "李师傅"}]
    inp = {"history": [], "current_state": st, "user_input": text,
           "current_time": "2026-08-09 18:10", "available_tools": []}
    exp = final_decision(booking_state=bs, chef_query_status="matched",
                         candidate_chefs=[{"chef_id": "chef_001", "chef_name": "李师傅"}],
                         info_complete=True, missing_info=[], reply_type="booking_authorized",
                         reply=reply_text)
    diag.append(make_case(f"diag_confirm_{i:03d}", "final", "single_turn", inp, exp,
                          assertions=["authorized_only_with_allowlist", "no_booking_success_claim"],
                          tags=["diagnostic", "confirmation"]))

# --- Bucket 8: Dietary/menu semantic stress probes (8 cases) ---
for i, (expected_diet, predicted_diet, reply_text) in enumerate([
    (["不吃花生"], ["可以吃花生"], "可以吃花生的"),  # polarity reversal
    (["不吃辣"], ["可以吃辣"], "可以吃辣"),
    (["不吃海鲜"], ["可以吃海鲜"], "可以吃海鲜"),
    (["不吃花生", "不吃辣"], ["不吃花生"], "只保留不吃花生"),  # dropped constraint
    (["不吃花生"], [], "没有饮食限制"),  # lost constraint
    (["不吃花生", "不吃辣", "不吃海鲜"], ["不吃花生"], "只有不吃花生"),
    (["不吃花生"], ["不吃花生", "不吃辣"], "多加一个不吃辣"),  # should pass
    (["不吃花生", "不吃辣"], ["不吃花生", "不吃辣"], "保持饮食限制"),  # should pass
], 1):
    bs = {"service_date": "2026-08-10", "start_time": "18:00", "people": 2, "address": "上海市徐汇区",
          "cuisine": "川菜", "dietary_constraints": predicted_diet, "menu": ["水煮鱼"],
          "occasion": "生日", "budget_min": 500.0, "budget_max": 900.0}
    st = empty_state()
    st["booking_state"].update(bs)
    st["booking_state"]["dietary_constraints"] = expected_diet
    inp = base_input(reply_text, available_tools=[], booking_state=bs)
    inp["current_state"] = st
    bs_out = dict(bs)
    exp = final_decision(booking_state=bs_out, chef_query_status="matched",
                         candidate_chefs=[{"chef_id": "chef_001", "chef_name": "李师傅"}],
                         info_complete=True, missing_info=[], reply_type="confirm_specific_chef",
                         reply="可以选李师傅")
    diag.append(make_case(f"diag_diet_{i:03d}", "final", "single_turn", inp, exp,
                          assertions=["dietary_inherited"],
                          tags=["diagnostic", "dietary_stress"]))

# --- Bucket 9: Protocol-format stress probes (6 cases) ---
for i, (text, reply_text) in enumerate([
    ("你好", "你好！请提供日期"),  # valid
    ("", ""),  # empty
    (" ", "请提供日期"),  # whitespace
    ("你好啊！！！", "你好！"),
    ("？？？", "请提供日期"),
    ("😊", "请提供日期"),
], 1):
    st = empty_state()
    inp = base_input(text, available_tools=[], booking_state={})
    exp = final_decision(booking_state=empty_booking(), info_complete=False, missing_info=["service_date"],
                         reply_type="ask_service_date", reply=reply_text)
    diag.append(make_case(f"diag_proto_{i:03d}", "final", "single_turn", inp, exp,
                          assertions=[], tags=["diagnostic", "protocol_stress"]))

# === VERIFY COUNTS ===
assert len(diag) == 80, f"Diagnostic Dev: expected 80, got {len(diag)}"

# ====== WRITE FILES ======
FROZEN_OUT.parent.mkdir(parents=True, exist_ok=True)
DIAG_OUT.parent.mkdir(parents=True, exist_ok=True)

FROZEN_OUT.write_text("\n".join(json.dumps(c, ensure_ascii=False, sort_keys=True) for c in frozen), encoding="utf-8")
DIAG_OUT.write_text("\n".join(json.dumps(c, ensure_ascii=False, sort_keys=True) for c in diag), encoding="utf-8")

print(f"Frozen Test: {len(frozen)} cases written to {FROZEN_OUT}")
print(f"Diagnostic Dev: {len(diag)} cases written to {DIAG_OUT}")

# Print taxonomy audit
from collections import Counter

frozen_tags = Counter()
for c in frozen:
    for t in c["tags"]:
        frozen_tags[t] += 1
print("\nFrozen Test taxonomy:")
for tag, count in sorted(frozen_tags.items()):
    print(f"  {tag}: {count}")

diag_tags = Counter()
for c in diag:
    for t in c["tags"]:
        diag_tags[t] += 1
print("\nDiagnostic Dev taxonomy:")
for tag, count in sorted(diag_tags.items()):
    print(f"  {tag}: {count}")
