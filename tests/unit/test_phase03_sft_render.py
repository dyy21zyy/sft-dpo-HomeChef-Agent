"""Task 5 Phase 03: SFT rendering tests."""

import json
from pathlib import Path

from homechef_booking.data.raw_validator import load_valid_raw_samples
from homechef_booking.data.sft_render import render_sft_sample
from homechef_booking.prompts import PromptBuilder


def test_sft_messages_use_prompt_builder_exactly():
    raw = load_valid_raw_samples(Path("tests/fixtures/datasets/raw_smoke_candidate.jsonl"))[0]
    sft = render_sft_sample(raw)
    expected_prompt = PromptBuilder().build_messages(raw.input)
    prompt_msg_count = len(expected_prompt)
    assert len(sft.messages) == prompt_msg_count + 1
    for i in range(prompt_msg_count):
        assert sft.messages[i].role == expected_prompt[i]["role"]
        assert sft.messages[i].content == str(expected_prompt[i]["content"] if expected_prompt[i].get("content") is not None else "")
    assert sft.messages[-1].role == "assistant"
    assert sft.raw_id == raw.id


def test_sft_assistant_uses_canonical_json():
    raw = load_valid_raw_samples(Path("tests/fixtures/datasets/raw_smoke_candidate.jsonl"))[0]
    sft = render_sft_sample(raw)
    canonical = json.dumps(raw.expected.model_dump(mode="json", exclude_none=False), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert sft.messages[-1].content == canonical


def test_sft_preserves_tool_call_canonical():
    raw = load_valid_raw_samples(Path("tests/fixtures/datasets/raw_smoke_candidate.jsonl"))[1]
    sft = render_sft_sample(raw)
    assert sft.messages[-1].role == "assistant"
    assert sft.raw_id == raw.id
    assert json.loads(sft.messages[-1].content)["action"] == "tool_call"
