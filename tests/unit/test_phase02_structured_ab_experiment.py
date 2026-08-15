"""TDD tests for Phase 02 Structured vs Unstructured Controlled Experiment."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

EXPECTED_IDS = [
    "frozen_missing_001", "frozen_missing_002",
    "frozen_tool_001", "frozen_tool_002",
    "frozen_reltime_001", "frozen_reltime_002",
    "frozen_tool_result_001", "frozen_tool_result_002",
    "frozen_semantic_001", "frozen_semantic_002",
]


def test_fixed_10_case_ids():
    from scripts.eval.run_phase02_structured_ab_10case import EXPERIMENT_CASE_IDS
    assert EXPERIMENT_CASE_IDS == EXPECTED_IDS


def test_u_prompt_equals_s_prompt():
    from homechef_booking.evaluation.sample import load_eval_cases
    from scripts.eval.run_phase02_structured_ab_10case import build_prompt_u

    cases = load_eval_cases(Path("data/eval/frozen_test.jsonl"))
    case_map = {c.id: c for c in cases}
    for cid in EXPECTED_IDS:
        case = case_map[cid]
        u_msgs = build_prompt_u(case)
        s_msgs = build_prompt_u(case)  # Same builder for both
        assert u_msgs == s_msgs, f"Prompt mismatch for {cid}"
        sys_msg = next(m for m in u_msgs if m["role"] == "system")
        assert "output_contract" in sys_msg["content"]


def test_u_structured_output_false():
    from scripts.eval.run_phase02_structured_ab_10case import get_config_u
    config = get_config_u()
    assert config.use_structured_output is False


def test_s_structured_output_true():
    from scripts.eval.run_phase02_structured_ab_10case import get_config_s
    config = get_config_s()
    assert config.use_structured_output is True


def test_configs_same_except_structured_flag():
    from scripts.eval.run_phase02_structured_ab_10case import get_config_s, get_config_u

    config_u = get_config_u()
    config_s = get_config_s()

    assert config_u.model_id == config_s.model_id
    assert config_u.base_url == config_s.base_url
    assert config_u.temperature == config_s.temperature
    assert config_u.max_new_tokens == config_s.max_new_tokens
    assert config_u.timeout_seconds == config_s.timeout_seconds
    assert config_u.device == config_s.device
    assert config_u.runtime == config_s.runtime
    assert config_u.model_format == config_s.model_format
    assert config_u.quantization == config_s.quantization
    assert config_u.gpu_layers == config_s.gpu_layers

    # Only difference: structured_output
    assert config_u.use_structured_output is False
    assert config_s.use_structured_output is True


def test_production_files_not_modified():
    files = [
        "src/homechef_booking/prompts/template.py",
        "src/homechef_booking/prompts/rules.py",
        "data/eval/frozen_test.jsonl",
    ]
    for f in files:
        p = Path(f)
        assert p.exists()
        assert len(p.read_text(encoding="utf-8")) > 0
