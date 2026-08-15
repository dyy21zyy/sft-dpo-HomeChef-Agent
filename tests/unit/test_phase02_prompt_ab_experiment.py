"""TDD tests for Phase 02 Prompt A/B Controlled Experiment."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root so we can import scripts.eval.*
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def test_experiment_10_cases_selected():
    from scripts.eval.run_phase02_prompt_ab_10case import select_10_cases
    selected = select_10_cases()
    assert len(selected) == 10

    from collections import Counter
    tag_counts = Counter()
    for case in selected:
        tags = set(case.tags)
        if "missing_required_slots" in tags:
            tag_counts["missing_required_slots"] += 1
        if "valid_search_tool_call" in tags:
            tag_counts["valid_search_tool_call"] += 1
        if "tool_result" in tags:
            tag_counts["tool_result"] += 1
        if "relative_time" in tags:
            tag_counts["relative_time"] += 1
        if "semantic_slots" in tags:
            tag_counts["semantic_slots"] += 1

    assert tag_counts["missing_required_slots"] == 2
    assert tag_counts["valid_search_tool_call"] == 2
    assert tag_counts["tool_result"] == 2
    assert tag_counts["relative_time"] == 2
    assert tag_counts["semantic_slots"] == 2


def test_build_prompt_a_uses_original_builder():
    from homechef_booking.evaluation.sample import load_eval_cases
    from scripts.eval.run_phase02_prompt_ab_10case import build_prompt_a

    cases = load_eval_cases(Path("data/eval/frozen_test.jsonl"))
    case = cases[0]
    messages = build_prompt_a(case)

    assert isinstance(messages, list)
    assert len(messages) > 0
    assert "role" in messages[0]
    assert "content" in messages[0]
    system_msg = next((m for m in messages if m["role"] == "system"), None)
    assert system_msg is not None
    assert "output_contract" in system_msg["content"]


def test_build_prompt_b_adds_strict_block_only():
    from homechef_booking.evaluation.sample import load_eval_cases
    from scripts.eval.run_phase02_prompt_ab_10case import build_prompt_a, build_prompt_b

    cases = load_eval_cases(Path("data/eval/frozen_test.jsonl"))
    case = cases[0]
    messages_a = build_prompt_a(case)
    messages_b = build_prompt_b(case)

    assert len(messages_b) == len(messages_a)
    for ma, mb in zip(messages_a, messages_b, strict=True):
        if ma["role"] != "system":
            assert ma == mb

    sys_a = next(m for m in messages_a if m["role"] == "system")
    sys_b = next(m for m in messages_b if m["role"] == "system")
    assert sys_a["content"] in sys_b["content"]
    assert "STRICT OUTPUT REQUIREMENTS" in sys_b["content"]
    assert "STRICT OUTPUT REQUIREMENTS" not in sys_a["content"]


def test_production_files_not_modified():
    files_to_check = [
        "src/homechef_booking/prompts/template.py",
        "src/homechef_booking/prompts/rules.py",
        "data/eval/frozen_test.jsonl",
    ]
    for f in files_to_check:
        p = Path(f)
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        assert len(content) > 0


def test_selected_cases_in_order():
    from homechef_booking.evaluation.sample import load_eval_cases
    from scripts.eval.run_phase02_prompt_ab_10case import select_10_cases

    selected = select_10_cases()
    ids = [c.id for c in selected]
    all_cases = load_eval_cases(Path("data/eval/frozen_test.jsonl"))
    all_ids = [c.id for c in all_cases]
    indices = [all_ids.index(cid) for cid in ids]
    assert indices == sorted(indices)


def test_no_structured_output():
    from scripts.eval.run_phase02_prompt_ab_10case import get_experiment_config
    config = get_experiment_config()
    assert config.use_structured_output is False


def test_strict_block_content():
    from scripts.eval.run_phase02_prompt_ab_10case import STRICT_OUTPUT_BLOCK
    block = STRICT_OUTPUT_BLOCK

    assert 'must be exactly one of the action values' in block.lower()
    assert 'must be exactly the literal string "tool_call"' in block.lower()
    assert 'never use "find_chefs" as the value of "action"' in block.lower()
    assert 'tool_call_required_keys' in block.lower()
    assert 'final_required_keys' in block.lower()
    assert 'even when a value is unknown or empty' in block.lower()
    assert 'must not be omitted merely because' in block.lower()
    assert 'do not invent a new reply_type' in block.lower()
    assert 'do not invent new action values' in block.lower()
    assert 'output exactly one json object' in block.lower()

    # Must NOT contain JSON examples
    assert '{' not in block
    assert '```' not in block
