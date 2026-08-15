"""Phase 04 shared SFT->LLaMA-Factory converter + derived dataset tests.

Covers the Codex review round on Tool Context:
- Deterministic context adaptation: delete empty assistant placeholder, preserve
  tool result + candidate-selection user, keep final assistant target unchanged.
- NO function_call fabrication from tool results.
- 1:1 derived dataset (540/60), no skip/drop/dedup/merge.
- Rows WITHOUT tool history keep equivalent training semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

from homechef_booking.training.dataset_adapter import (
    build_sft_derived_dataset,
    convert_sft_row_to_llamafactory_format,
)

_SFT_TRAIN = Path("data/processed/sft/v0.3/train.jsonl")
_SFT_VAL = Path("data/processed/sft/v0.3/val.jsonl")


def _sys() -> dict:
    return {"role": "system", "content": "booking contract"}


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _asst(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _tool(content: dict | str) -> dict:
    return {"role": "tool", "content": content}


# ── Tool-context trajectory (the review's exact case) ─────────────────────────


def _trajectory() -> dict:
    return {
        "messages": [
            _sys(),
            _user("8月30日我家要请客，地址武汉市东湖高新区，3位，上午12点开始，想吃粤菜，预算799到1126元。"),
            _asst(""),
            _tool({"mode": "search", "status": "matched", "candidates": [{"chef_id": "chef_0620"}]}),
            _user("第三位看起来不错，就选袁航师傅吧。"),
            _asst('{"action":"final","booking_state":{}}'),
        ]
    }


def test_tool_trajectory_no_empty_assistant():
    out = convert_sft_row_to_llamafactory_format(_trajectory())
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["system", "user", "assistant"]
    # No empty assistant placeholder anywhere.
    assert all(m.get("content", "") != "" or m["role"] != "assistant" for m in out["messages"])


def test_tool_trajectory_preserves_tool_result():
    out = convert_sft_row_to_llamafactory_format(_trajectory())
    user_content = out["messages"][1]["content"]
    assert "[TOOL_RESULT]" in user_content
    assert '"mode": "search"' in user_content
    assert "chef_0620" in user_content


def test_tool_trajectory_preserves_followup_selection():
    out = convert_sft_row_to_llamafactory_format(_trajectory())
    user_content = out["messages"][1]["content"]
    assert "[USER_FOLLOWUP]" in user_content
    assert "第三位看起来不错" in user_content
    assert "袁航" in user_content


def test_tool_trajectory_final_target_unchanged():
    out = convert_sft_row_to_llamafactory_format(_trajectory())
    assert out["messages"][-1]["content"] == '{"action":"final","booking_state":{}}'


def test_no_function_call_fabrication():
    # The empty assistant + tool result must NOT become a fake function_call.
    out = convert_sft_row_to_llamafactory_format(_trajectory())
    text = json.dumps(out, ensure_ascii=False)
    assert "function_call" not in text
    assert "tool_calls" not in text
    assert "arguments" not in text


# ── No-tool rows keep equivalent semantics ────────────────────────────────────


def test_simple_row_system_user_assistant():
    row = {"messages": [_sys(), _user("hello"), _asst("hi there")]}
    out = convert_sft_row_to_llamafactory_format(row)
    assert [m["role"] for m in out["messages"]] == ["system", "user", "assistant"]
    assert out["messages"][-1]["content"] == "hi there"
    assert out["messages"][1]["content"] == "hello"


def test_simple_row_no_tool_markers():
    row = {"messages": [_sys(), _user("hello"), _asst("hi")]}
    out = convert_sft_row_to_llamafactory_format(row)
    assert "[TOOL_RESULT]" not in out["messages"][1]["content"]
    assert "[USER_FOLLOWUP]" not in out["messages"][1]["content"]


# ── Other tool trajectories ───────────────────────────────────────────────────


def test_tool_result_directly_to_final():
    row = {"messages": [_sys(), _user("find chefs"), _asst(""),
                        _tool({"mode": "search", "status": "no_match"}),
                        _asst('{"action":"final","booking_state":{}}')]}
    out = convert_sft_row_to_llamafactory_format(row)
    assert "[TOOL_RESULT]" in out["messages"][1]["content"]
    assert '"status": "no_match"' in out["messages"][1]["content"]
    assert out["messages"][-1]["content"] == '{"action":"final","booking_state":{}}'


def test_tool_result_user_confirmation_to_final():
    row = {"messages": [_sys(), _user("book"), _asst(""), _tool({"mode": "unavailable"}),
                        _user("确认"), _asst('{"action":"final"}')]}
    out = convert_sft_row_to_llamafactory_format(row)
    assert "[USER_FOLLOWUP]" in out["messages"][1]["content"]
    assert "确认" in out["messages"][1]["content"]
    assert out["messages"][-1]["content"] == '{"action":"final"}'


# ── Derived dataset 1:1 ──────────────────────────────────────────────────────


def test_derived_train_540_rows(tmp_path: Path):
    manifest = build_sft_derived_dataset(_SFT_TRAIN, tmp_path / "derived_train.jsonl")
    assert manifest["source_rows"] == 540
    assert manifest["derived_rows"] == 540
    lines = [line for line in (tmp_path / "derived_train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 540


def test_derived_val_60_rows(tmp_path: Path):
    manifest = build_sft_derived_dataset(_SFT_VAL, tmp_path / "derived_val.jsonl")
    assert manifest["source_rows"] == 60
    assert manifest["derived_rows"] == 60


def test_derived_one_to_one_mapping(tmp_path: Path):
    manifest = build_sft_derived_dataset(_SFT_TRAIN, tmp_path / "d.jsonl")
    # 1:1: every source index maps to exactly one derived row (no drop/dedup/merge).
    assert len(manifest["mapping"]) == 540
    assert len(set(manifest["mapping"])) == 540


def test_derived_preserves_tool_context_count(tmp_path: Path):
    build_sft_derived_dataset(_SFT_TRAIN, tmp_path / "d_train.jsonl")
    derived = [json.loads(line) for line in (tmp_path / "d_train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    with_tool = sum(1 for d in derived if "[TOOL_RESULT]" in d["messages"][1]["content"])
    # All 226 train tool-context rows preserved in derived.
    assert with_tool == 226


def test_derived_val_preserves_tool_context_count(tmp_path: Path):
    build_sft_derived_dataset(_SFT_VAL, tmp_path / "d_val.jsonl")
    derived = [json.loads(line) for line in (tmp_path / "d_val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    with_tool = sum(1 for d in derived if "[TOOL_RESULT]" in d["messages"][1]["content"])
    assert with_tool == 24
