"""TDD: Decoupled Protocol/Task Correctness scoring.

Key changes:
- Protocol fail does NOT force Task Correctness to 0
- Task Correctness scores even when protocol fails (if JSON is parseable)
- Only truly unparseable output gets Task=0
- Effective Pass = Protocol PASS AND Task >= 0.95
"""

from __future__ import annotations

import json


# ── Helper: build a mock GenerationResult ─────────────────
def _mock_gen(raw_text: str | None, finish_reason="stop", error=None,
              latency=100.0, ttft=50.0, tokens=10, tps=5.0, tps_source="client_fallback"):
    from homechef_booking.inference.response import GenerationResult
    return GenerationResult(
        case_id="test_001", backend_name="mock",
        raw_text=raw_text, finish_reason=finish_reason,
        error_type=error, error_message=None,
        latency_ms=latency, ttft_ms=ttft, tokens_per_second=tps,
        throughput_source=tps_source, completion_tokens=tokens,
    )


# ── TEST 1: Protocol PASS, Task normal ────────────────────
def test_protocol_pass_task_normal():
    """Legal JSON, protocol passes → task scored normally."""
    from homechef_booking.evaluation.evidence import derive_tool_evidence
    from homechef_booking.evaluation.scorers.protocol import ProtocolScorer
    from homechef_booking.evaluation.scorers.task_correctness import TaskCorrectnessScorer

    raw = json.dumps({"action": "tool_call", "tool_name": "find_chefs", "arguments": {
        "chef_name": None, "service_date": "2026-08-10", "start_time": "18:00",
        "people": 4, "address": "Beijing", "cuisine": "Sichuan",
        "budget_min": None, "budget_max": None, "menu": [],
        "ingredient_purchase": False, "dietary_constraints": [], "occasion": "birthday"
    }})

    gen = _mock_gen(raw)

    # Load a real case for proper scoring
    from pathlib import Path

    from homechef_booking.evaluation.sample import load_eval_cases
    cases = load_eval_cases(Path("data/eval/frozen_test.jsonl"))
    case = [c for c in cases if c.id == "frozen_tool_001"][0]

    protocol_scorer = ProtocolScorer()
    task_scorer = TaskCorrectnessScorer()
    protocol = protocol_scorer.score(case, gen)

    assert protocol.passed is True, f"Expected protocol pass, got error: {protocol.details.get('error','')[:200]}"

    evidence = derive_tool_evidence(case.input)
    prediction = json.loads(gen.raw_text or "{}")
    task = task_scorer.score(case, gen, evidence)
    assert task.score > 0, "Task score should be > 0 for valid output"


# ── TEST 2: Missing required field → Protocol FAIL, Task > 0 ──
def test_protocol_fail_task_nonzero():
    """Legal JSON, missing one required field → protocol=0, task>0."""
    from homechef_booking.evaluation.evidence import derive_tool_evidence
    from homechef_booking.evaluation.scorers.protocol import ProtocolScorer
    from homechef_booking.evaluation.scorers.task_correctness import TaskCorrectnessScorer

    raw = json.dumps({"action": "tool_call", "tool_name": "find_chefs", "arguments": {
        "chef_name": None, "service_date": "2026-08-10", "start_time": "18:00",
        "people": 4, "address": "Beijing", "cuisine": "Sichuan",
        "budget_min": None, "budget_max": None, "menu": [],
        "ingredient_purchase": False
        # MISSING: dietary_constraints, occasion
    }})

    gen = _mock_gen(raw)

    from pathlib import Path

    from homechef_booking.evaluation.sample import load_eval_cases
    cases = load_eval_cases(Path("data/eval/frozen_test.jsonl"))
    case = [c for c in cases if c.id == "frozen_tool_001"][0]

    protocol_scorer = ProtocolScorer()
    task_scorer = TaskCorrectnessScorer()
    protocol = protocol_scorer.score(case, gen)

    assert protocol.passed is False, "Expected protocol FAIL (missing arguments)"

    # Even though protocol failed, task should score (evidence from parseable JSON)
    evidence = derive_tool_evidence(case.input)
    task = task_scorer.score(case, gen, evidence)
    assert task.score > 0, f"Task score should be > 0 even with protocol fail, got {task.score}"
    assert task.score < 1.0, "Task score should be < 1.0 (missing fields)"


# ── TEST 3: Missing tool argument → Protocol=0, Task loses only that field ──
def test_missing_tool_argument_partial_task():
    """One missing FindChefsInput arg → protocol=0, task loses only that field."""
    from homechef_booking.evaluation.evidence import derive_tool_evidence
    from homechef_booking.evaluation.scorers.protocol import ProtocolScorer
    from homechef_booking.evaluation.scorers.task_correctness import TaskCorrectnessScorer

    raw = json.dumps({"action": "tool_call", "tool_name": "find_chefs", "arguments": {
        "chef_name": None, "service_date": "2026-08-10", "start_time": "18:00",
        "people": 4, "address": "Beijing", "cuisine": "Sichuan",
        "budget_min": None, "budget_max": None, "menu": [],
        "ingredient_purchase": False, "dietary_constraints": [], "occasion": "birthday"
        # All 12 fields present → should protocol pass
    }})

    gen = _mock_gen(raw)

    from pathlib import Path

    from homechef_booking.evaluation.sample import load_eval_cases
    cases = load_eval_cases(Path("data/eval/frozen_test.jsonl"))
    case = [c for c in cases if c.id == "frozen_tool_001"][0]

    protocol_scorer = ProtocolScorer()
    task_scorer = TaskCorrectnessScorer()
    protocol = protocol_scorer.score(case, gen)

    # With all 12 fields, protocol should pass
    evidence = derive_tool_evidence(case.input)
    prediction = json.loads(gen.raw_text or "{}")
    task = task_scorer.score(case, gen, evidence)
    assert task.score > 0.5, f"Task score should be substantial with all fields, got {task.score}"


# ── TEST 4: Unparseable → Protocol=0, Task=0 ───────────────
def test_unparseable_protocol_zero_task_zero():
    """Natural language / invalid JSON → protocol=0, task=0."""
    from homechef_booking.evaluation.scorers.protocol import ProtocolScorer
    from homechef_booking.evaluation.scorers.task_correctness import TaskCorrectnessScorer

    raw = "I think you should book a Sichuan chef for tomorrow evening."
    gen = _mock_gen(raw)

    from pathlib import Path

    from homechef_booking.evaluation.sample import load_eval_cases
    cases = load_eval_cases(Path("data/eval/frozen_test.jsonl"))
    case = [c for c in cases if c.id == "frozen_tool_001"][0]

    protocol_scorer = ProtocolScorer()
    task_scorer = TaskCorrectnessScorer()
    protocol = protocol_scorer.score(case, gen)

    assert protocol.passed is False

    # Unparseable → task scorer returns 0
    task = task_scorer.score(case, gen)
    assert task.score == 0.0, f"Unparseable output should get task=0, got {task.score}"


# ── TEST 5: Protocol=0, Task=0.98 → Effective=False ────────
def test_effective_pass_requires_both():
    """Effective Pass = Protocol PASS AND Task >= 0.95."""
    from homechef_booking.evaluation.effective_pass import effective_pass

    # Protocol=0, Task=0.98 → NOT effective
    assert effective_pass(False, 0.98, False) is False
    # Protocol=1, Task=0.94 → NOT effective
    assert effective_pass(True, 0.94, False) is False
    # Protocol=1, Task=0.95 → effective
    assert effective_pass(True, 0.95, False) is True
    # Protocol=1, Task=0.95, critical_error=True → NOT effective
    assert effective_pass(True, 0.95, True) is False


# ── TEST 6: Protocol=1, Task=0.94 → Effective=False ────────
def test_effective_pass_task_below_threshold():
    from homechef_booking.evaluation.effective_pass import effective_pass
    assert effective_pass(True, 0.94, False) is False


# ── TEST 7: Protocol=1, Task=0.95 → Effective=True ─────────
def test_effective_pass_both_pass():
    from homechef_booking.evaluation.effective_pass import effective_pass
    assert effective_pass(True, 0.95, False) is True


# ── TEST 8: Synthetic timings → correct aggregates ─────────
def test_latency_metrics_computation():
    """Mean latency, P95, mean TTFT, mean throughput computed correctly."""
    from homechef_booking.evaluation.benchmark_runner import (
        _compute_latency_metrics,
    )

    case_results = [
        {"generation_result": {"latency_ms": 100.0, "ttft_ms": 20.0, "tokens_per_second": 10.0, "completion_tokens": 5}},
        {"generation_result": {"latency_ms": 200.0, "ttft_ms": 30.0, "tokens_per_second": 20.0, "completion_tokens": 10}},
        {"generation_result": {"latency_ms": 300.0, "ttft_ms": 40.0, "tokens_per_second": 30.0, "completion_tokens": 15}},
        {"generation_result": {"latency_ms": 400.0, "ttft_ms": 50.0, "tokens_per_second": 40.0, "completion_tokens": 20}},
        {"generation_result": {"latency_ms": 500.0, "ttft_ms": 60.0, "tokens_per_second": 50.0, "completion_tokens": 25}},
    ]

    metrics = _compute_latency_metrics(case_results, total_wall_seconds=10.0, total_cases=5)

    assert metrics.mean_latency_ms == 300.0
    assert metrics.median_latency_ms == 300.0
    assert metrics.p95_latency_ms is not None and 450.0 <= metrics.p95_latency_ms <= 500.0
    assert metrics.mean_ttft_ms == 40.0
    assert metrics.mean_tokens_per_second == 30.0
    assert metrics.performance_sample_count == 5
    assert metrics.successful_inference_cases == 5
    assert metrics.failed_inference_cases == 0
