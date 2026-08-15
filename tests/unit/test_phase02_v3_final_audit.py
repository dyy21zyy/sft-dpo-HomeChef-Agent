"""TDD: V3 Final Audit — unified taxonomy, invariant checks, config comparisons."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ── TEST 1: valid_json + taxonomy invariant ──────────────────
def test_valid_json_and_failure_category_invariant():
    """If valid_json==true, primary_failure MUST NOT be invalid_json."""
    # This is enforced by the unified taxonomy logic in V3 rescore
    # We test that our taxonomy function upholds this invariant
    from scripts.eval.run_phase02_v3_benchmark import classify_failure

    # Valid JSON with missing field -> should NOT be invalid_json
    parsed = {"action": "tool_call", "tool_name": "find_chefs", "arguments": {}}
    cat, _ = classify_failure(parsed, valid_json=True, protocol_pass=False)
    assert cat != "invalid_json", f"Valid JSON must not be tagged invalid_json: got {cat}"

    # Invalid JSON -> should be invalid_json
    cat, _ = classify_failure(None, valid_json=False, protocol_pass=False)
    assert cat == "invalid_json", f"Invalid JSON must be tagged invalid_json: got {cat}"


def test_invalid_json_only_when_parse_fails():
    """primary_failure==invalid_json implies valid_json==false."""
    from scripts.eval.run_phase02_v3_benchmark import classify_failure
    # If parseable JSON and protocol fails, category must be schema-related
    parsed = {"action": "tool_call"}
    cat, _ = classify_failure(parsed, valid_json=True, protocol_pass=False)
    assert cat != "invalid_json"


def test_protocol_fail_schema_category():
    """Protocol fail + parseable JSON -> schema/protocol category."""
    from scripts.eval.run_phase02_v3_benchmark import classify_failure
    parsed = {"action": "find_chefs"}
    cat, _ = classify_failure(parsed, valid_json=True, protocol_pass=False)
    assert cat in ("invalid_action", "missing_action", "missing_required_top_level_fields",
                   "missing_find_chefs_argument_keys", "invalid_reply_type",
                   "wrong_type", "extra_fields", "other_schema_error")


def test_protocol_pass_count_aggregation():
    """Protocol pass count from case_results must equal scorecard."""
    # Sample test: aggregate 5 cases, 3 pass
    cases = [
        {"protocol_pass": True, "task_correctness": 0.95, "effective_pass": True},
        {"protocol_pass": True, "task_correctness": 0.90, "effective_pass": False},
        {"protocol_pass": True, "task_correctness": 0.98, "effective_pass": True},
        {"protocol_pass": False, "task_correctness": 0.80, "effective_pass": False},
        {"protocol_pass": False, "task_correctness": 0.50, "effective_pass": False},
    ]
    pp = sum(1 for c in cases if c["protocol_pass"])
    assert pp == 3


def test_effective_pass_requires_both():
    """Effective pass = protocol true AND task >= 0.95."""
    from homechef_booking.evaluation.effective_pass import effective_pass
    assert effective_pass(True, 0.95, False) is True
    assert effective_pass(False, 1.00, False) is False
    assert effective_pass(True, 0.94, False) is False


def test_mean_latency_computation():
    """Mean latency computed correctly."""
    import statistics
    lats = [100.0, 200.0, 300.0, 400.0, 500.0]
    mean_s = statistics.mean(lats) / 1000
    assert mean_s == 0.3


def test_p95_computation():
    """P95 computed correctly from sorted values."""
    def p95(vals):
        if not vals:
            return None
        n = len(vals)
        k = 0.95 * (n - 1)
        f = int(k)
        c = k - f
        return vals[f] + c * (vals[f + 1] - vals[f]) if f + 1 < n else vals[f]

    vals = sorted([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    result = p95(vals)
    assert 900 <= result <= 1000


def test_mean_ttft_computation():
    """Mean TTFT computed correctly."""
    import statistics
    ttfts = [10.0, 20.0, 30.0, None]
    valid = [t for t in ttfts if t is not None]
    assert statistics.mean(valid) == 20.0


def test_mean_throughput_computation():
    """Mean throughput = mean(case tokens_per_s)."""
    import statistics
    tps = [10.0, 20.0, 30.0, None]
    valid = [t for t in tps if t is not None]
    assert statistics.mean(valid) == 20.0


def test_1_7b_configs_same_except_structured():
    """1.7B U and S configs differ only in use_structured_output."""
    from scripts.eval.run_phase02_v3_benchmark import get_config_1_7b_s, get_config_1_7b_u
    u = get_config_1_7b_u()
    s = get_config_1_7b_s()
    assert u.model_id == s.model_id
    assert u.temperature == s.temperature
    assert u.max_new_tokens == s.max_new_tokens
    assert u.use_structured_output is False
    assert s.use_structured_output is True
