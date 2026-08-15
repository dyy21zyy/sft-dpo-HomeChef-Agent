"""Phase 02 V3 Final: 2x2 Controlled Benchmark + Unified Taxonomy + Semantic Analysis.

Runs:
  A. 1.7B Unstructured (120 cases)
  B. 1.7B Structured   (120 cases)
  C. 4B  Unstructured  (120 cases)
  D. 4B  Structured    (120 cases)

Unified failure taxonomy, proper valid_json invariant, full performance metrics.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from homechef_booking.evaluation.evidence import derive_tool_evidence
from homechef_booking.evaluation.sample import EvalCase
from homechef_booking.evaluation.scorers.protocol import ProtocolScorer
from homechef_booking.evaluation.scorers.task_correctness import TaskCorrectnessScorer
from homechef_booking.inference.backend import GenerationParams
from homechef_booking.inference.llama_cpp_backend import LlamaCppServerBackend, LlamaCppServerConfig
from homechef_booking.inference.runner import run_inference
from homechef_booking.schemas.booking import ReplyType

CANONICAL_RT = sorted(e.value for e in ReplyType)


# ── Configs ──────────────────────────────────────────────
def get_config_1_7b_u() -> LlamaCppServerConfig:
    return LlamaCppServerConfig(
        model_id="Qwen/Qwen3-1.7B-Base", base_url="http://127.0.0.1:8080/v1",
        max_new_tokens=512, temperature=0.0, timeout_seconds=600,
        use_structured_output=False,
        device="cpu", runtime="llama.cpp", model_format="gguf",
        quantization="Q8_0", gpu_layers=0,
    )

def get_config_1_7b_s() -> LlamaCppServerConfig:
    return LlamaCppServerConfig(
        model_id="Qwen/Qwen3-1.7B-Base", base_url="http://127.0.0.1:8080/v1",
        max_new_tokens=512, temperature=0.0, timeout_seconds=600,
        use_structured_output=True,
        device="cpu", runtime="llama.cpp", model_format="gguf",
        quantization="Q8_0", gpu_layers=0,
    )

def get_config_4b_u() -> LlamaCppServerConfig:
    return LlamaCppServerConfig(
        model_id="Qwen/Qwen3-4B-Instruct-2507", base_url="http://127.0.0.1:8080/v1",
        max_new_tokens=512, temperature=0.0, timeout_seconds=900,
        use_structured_output=False,
        device="cpu", runtime="llama.cpp", model_format="gguf",
        quantization="Q4_K_M", gpu_layers=0,
    )

def get_config_4b_s() -> LlamaCppServerConfig:
    return LlamaCppServerConfig(
        model_id="Qwen/Qwen3-4B-Instruct-2507", base_url="http://127.0.0.1:8080/v1",
        max_new_tokens=512, temperature=0.0, timeout_seconds=900,
        use_structured_output=True,
        device="cpu", runtime="llama.cpp", model_format="gguf",
        quantization="Q4_K_M", gpu_layers=0,
    )


# ── Unified Failure Taxonomy ────────────────────────────
def classify_failure(parsed: dict | None, valid_json: bool, protocol_pass: bool) -> tuple[str, list[str]]:
    """Returns (primary_failure, multi_label_tags).

    Invariant: if valid_json==true, primary_failure != "invalid_json".
    """
    if protocol_pass:
        return "none", []
    if not valid_json:
        return "invalid_json", ["invalid_json"]

    action = parsed.get("action", "MISSING")

    if action not in ("tool_call", "final"):
        return "invalid_action", ["invalid_action"]

    if action == "tool_call":
        req = ["action", "tool_name", "arguments"]
        miss = [k for k in req if k not in parsed]
        if miss:
            return "missing_required_top_level_fields", ["missing_required_top_level_fields"] + miss

        args = parsed.get("arguments", {})
        if not isinstance(args, dict):
            return "wrong_type", ["wrong_type"]

        fci = ["chef_name","service_date","start_time","people","address","cuisine",
               "budget_min","budget_max","menu","ingredient_purchase","dietary_constraints","occasion"]
        miss_args = [k for k in fci if k not in args]
        if miss_args:
            return "missing_find_chefs_argument_keys", ["missing_find_chefs_argument_keys"] + miss_args

        return "other_schema_error", ["other_schema_error"]

    elif action == "final":
        req = ["action","booking_state","chef_query_status","candidate_chefs",
               "info_complete","unrelated","missing_info","reply_type","reply"]
        miss = [k for k in req if k not in parsed]
        if miss:
            return "missing_required_top_level_fields", ["missing_required_top_level_fields"] + miss

        rt = parsed.get("reply_type")
        if rt not in CANONICAL_RT:
            return "invalid_reply_type", ["invalid_reply_type"]

        return "other_schema_error", ["other_schema_error"]

    return "other_schema_error", ["other_schema_error"]


# ── Single case runner ──────────────────────────────────
def run_case(case: EvalCase, config: LlamaCppServerConfig, label: str) -> dict:
    backend = LlamaCppServerBackend()
    backend.load(config)
    params = GenerationParams()
    gen = run_inference(case.id, case.input, backend, params)

    protocol_scorer = ProtocolScorer()
    task_scorer = TaskCorrectnessScorer()
    protocol = protocol_scorer.score(case, gen)

    try:
        evidence = derive_tool_evidence(case.input)
    except Exception:
        evidence = None

    parseable = False
    pred = {}
    try:
        pred = json.loads(gen.raw_text or "")
        parseable = isinstance(pred, dict)
    except json.JSONDecodeError:
        pass

    if parseable:
        task = task_scorer.score(case, gen, evidence)
    else:
        task = task_scorer.score(case, gen, None)

    eff = protocol.passed is True and task.score >= 0.95
    primary_failure, multi_tags = classify_failure(
        pred if parseable else None, parseable, protocol.passed is True)

    return {
        "case_id": case.id, "tags": case.tags, "output_kind": case.output_kind,
        "label": label,
        "raw_text": gen.raw_text,
        "parsed_json": pred if parseable else None,
        "valid_json": parseable,
        "protocol_pass": protocol.passed is True,
        "protocol_error": str(protocol.details.get("error",""))[:200] if not protocol.passed else "",
        "task_correctness": task.score,
        "structured_score": task.details.get("structured_score", 0.0),
        "reply_score": task.details.get("reply_score", 0.0),
        "effective_pass": eff,
        "primary_failure": primary_failure,
        "multi_label_tags": multi_tags,
        "action": pred.get("action") if parseable else None,
        "reply_type": pred.get("reply_type") if parseable else None,
        "latency_ms": gen.latency_ms, "ttft_ms": gen.ttft_ms,
        "completion_tokens": gen.completion_tokens,
        "tokens_per_second": gen.tokens_per_second,
        "throughput_source": gen.throughput_source,
        "finish_reason": gen.finish_reason,
    }


# ── Aggregate ───────────────────────────────────────────
def aggregate(results: list[dict], label: str) -> dict:
    n = len(results)
    pp = sum(1 for r in results if r["protocol_pass"])
    ep = sum(1 for r in results if r["effective_pass"])
    vj = sum(1 for r in results if r["valid_json"])
    tc = statistics.mean([r["task_correctness"] for r in results])
    ss = statistics.mean([r["structured_score"] for r in results])
    rs = statistics.mean([r["reply_score"] for r in results])

    pfht = [r for r in results if not r["protocol_pass"] and r["task_correctness"] >= 0.80]
    fails = Counter(r["primary_failure"] for r in results if not r["protocol_pass"])
    multi_tags = Counter()
    for r in results:
        for t in r.get("multi_label_tags", []):
            multi_tags[t] += 1

    # Scenario slices
    sg = defaultdict(list)
    for r in results:
        for t in r["tags"]:
            sg[t].append(r)
    slices = {}
    for t, g in sorted(sg.items()):
        slices[t] = {"count": len(g), "mean_task": statistics.mean([r["task_correctness"] for r in g]),
                      "effective_pass": sum(1 for r in g if r["effective_pass"])}

    # Performance
    lats = [r["latency_ms"] for r in results if r["latency_ms"] and r["latency_ms"] > 0]
    lats_sorted = sorted(lats)
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] and r["ttft_ms"] > 0]
    tpses = [r["tokens_per_second"] for r in results if r["tokens_per_second"] and r["tokens_per_second"] > 0]

    def _p95(vals):
        if not vals:
            return None
        n2 = len(vals)
        k = 0.95 * (n2 - 1)
        f = int(k)
        c = k - f
        return vals[f] + c * (vals[f + 1] - vals[f]) if f + 1 < n2 else vals[f]

    return {
        "label": label, "total": n,
        "protocol_pass": pp, "protocol_pass_rate": pp/n,
        "effective_pass": ep, "effective_pass_rate": ep/n,
        "valid_json": vj, "valid_json_rate": vj/n,
        "mean_task_correctness": tc, "mean_structured_score": ss, "mean_reply_score": rs,
        "protocol_fail_high_task": {"count": len(pfht), "rate": len(pfht)/n,
            "case_ids": [r["case_id"] for r in pfht],
            "primary_failures": dict(Counter(r["primary_failure"] for r in pfht))},
        "primary_failures": dict(fails),
        "multi_label_tags": dict(multi_tags),
        "scenario_slices": slices,
        "performance": {
            "mean_latency_s": round(statistics.mean(lats)/1000,3) if lats else None,
            "p50_latency_s": round(statistics.median(lats)/1000,3) if lats else None,
            "p95_latency_s": round(_p95(lats_sorted)/1000,3) if lats_sorted else None,
            "min_latency_s": round(min(lats)/1000,3) if lats else None,
            "max_latency_s": round(max(lats)/1000,3) if lats else None,
            "mean_ttft_s": round(statistics.mean(ttfts)/1000,3) if ttfts else None,
            "p95_ttft_s": round(_p95(sorted(ttfts))/1000,3) if ttfts else None,
            "mean_tokens_per_second": round(statistics.mean(tpses),2) if tpses else None,
            "mean_output_tokens": round(statistics.mean([r["completion_tokens"] for r in results if r["completion_tokens"]]),1),
            "latency_sample_count": len(lats),
        },
    }


# ── Runtime manifest ────────────────────────────────────
def runtime_manifest(config: LlamaCppServerConfig) -> dict:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
        "model_id": config.model_id,
        "quantization": config.quantization,
        "temperature": config.temperature,
        "max_new_tokens": config.max_new_tokens,
        "use_structured_output": config.use_structured_output,
        "base_url": config.base_url,
        "gpu_layers": config.gpu_layers,
        "frozen_test_sha256": hashlib.sha256(
            Path("data/eval/frozen_test.jsonl").read_bytes()).hexdigest(),
    }


# ── Semantic failure analysis ───────────────────────────
def semantic_failure_analysis(results: list[dict]) -> dict:
    """Analyze effective_fail cases with detailed semantic error breakdown."""
    eff_fail = [r for r in results if r["protocol_pass"] and r["task_correctness"] < 0.95]

    # Error occurrence (multi-label)
    error_occurrence = Counter()
    for r in eff_fail:
        structured = r["structured_score"]
        reply = r["reply_score"]
        if structured < 1.0:
            error_occurrence["structured_error"] += 1
        if reply < 1.0:
            error_occurrence["reply_error"] += 1
        # Categorize by scenario
        for tag in r["tags"]:
            error_occurrence[f"scenario:{tag}"] += 1

    primary = Counter()

    for r in eff_fail:
        tags = set(r["tags"])
        tc = r["task_correctness"]

        if "tool_timing" in str(r.get("parsed_json", {})):
            primary["tool_timing"] += 1
        elif tc < 0.50:
            primary["other"] += 1
        elif tc < 0.80:
            # Categorize by scenario
            if "relative_time" in tags:
                primary["relative_time"] += 1
            elif "tool_result" in tags:
                primary["tool_fact_grounding"] += 1
            elif "confirmation" in tags:
                primary["confirmation"] += 1
            else:
                primary["slot_extraction"] += 1
        else:
            # tc >= 0.80 but < 0.95 — minor errors
            if "tool_result" in tags:
                primary["reply_type"] += 1
            elif "confirmation" in tags:
                primary["confirmation"] += 1
            elif "relative_time" in tags:
                primary["relative_time"] += 1
            else:
                primary["other"] += 1

    # Task score buckets
    buckets = {"<0.50": 0, "0.50-0.80": 0, "0.80-0.95": 0}
    for r in eff_fail:
        tc = r["task_correctness"]
        if tc < 0.50:
            buckets["<0.50"] += 1
        elif tc < 0.80:
            buckets["0.50-0.80"] += 1
        else:
            buckets["0.80-0.95"] += 1

    return {
        "effective_fail_count": len(eff_fail),
        "error_occurrence": dict(error_occurrence.most_common()),
        "primary_semantic_failure": dict(primary),
        "primary_total": sum(primary.values()),
        "task_score_buckets": buckets,
    }
