"""Phase 02 Structured vs Unstructured Controlled Experiment.

10 cases, 4B-Instruct-2507, 2 modes = 20 generations.

U = use_structured_output=false (baseline)
S = use_structured_output=true  (GBNF constrained decoding)

Same PromptBuilder, same scorer, same cases, same temperature=0.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from homechef_booking.evaluation.evidence import derive_tool_evidence
from homechef_booking.evaluation.sample import EvalCase, load_eval_cases
from homechef_booking.evaluation.scorers.protocol import ProtocolScorer
from homechef_booking.evaluation.scorers.task_correctness import TaskCorrectnessScorer
from homechef_booking.inference.backend import GenerationParams
from homechef_booking.inference.llama_cpp_backend import LlamaCppServerConfig
from homechef_booking.inference.runner import run_inference
from homechef_booking.prompts.template import PromptBuilder
from homechef_booking.schemas.booking import ReplyType

CANONICAL_REPLY_TYPES = sorted(e.value for e in ReplyType)

EXPERIMENT_CASE_IDS = [
    "frozen_missing_001", "frozen_missing_002",
    "frozen_tool_001", "frozen_tool_002",
    "frozen_reltime_001", "frozen_reltime_002",
    "frozen_tool_result_001", "frozen_tool_result_002",
    "frozen_semantic_001", "frozen_semantic_002",
]


def get_config_u() -> LlamaCppServerConfig:
    return LlamaCppServerConfig(
        base_url="http://127.0.0.1:8080/v1",
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        max_new_tokens=512,
        temperature=0.0,
        timeout_seconds=900,
        use_structured_output=False,
        device="cpu", runtime="llama.cpp",
        model_format="gguf", quantization="Q4_K_M",
        gpu_layers=0,
    )


def get_config_s() -> LlamaCppServerConfig:
    return LlamaCppServerConfig(
        base_url="http://127.0.0.1:8080/v1",
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        max_new_tokens=512,
        temperature=0.0,
        timeout_seconds=900,
        use_structured_output=True,
        device="cpu", runtime="llama.cpp",
        model_format="gguf", quantization="Q4_K_M",
        gpu_layers=0,
    )


def build_prompt_u(case: EvalCase) -> list[dict]:
    builder = PromptBuilder()
    return builder.build_messages(case.input)


def run_single_case(
    case: EvalCase, backend_config: LlamaCppServerConfig, label: str
) -> dict:
    from homechef_booking.inference.llama_cpp_backend import LlamaCppServerBackend
    backend = LlamaCppServerBackend()
    backend.load(backend_config)
    params = GenerationParams()
    gen = run_inference(case.id, case.input, backend, params)

    protocol_scorer = ProtocolScorer()
    task_scorer = TaskCorrectnessScorer()

    protocol = protocol_scorer.score(case, gen)
    if protocol.passed is True:
        evidence = derive_tool_evidence(case.input)
        task = task_scorer.score(case, gen, evidence)
        eff_pass = task.score >= 0.95
    else:
        task = task_scorer.score(case, gen)
        eff_pass = False

    raw = gen.raw_text or ""
    json_str = re.sub(r'^```(?:json)?\s*\n?', '', raw.strip())
    json_str = re.sub(r'\n?```\s*$', '', json_str)
    try:
        parsed = json.loads(json_str)
        valid_json = True
    except json.JSONDecodeError:
        parsed = None
        valid_json = False

    failure_category = "none"
    missing_fields: list[str] = []
    if not protocol.passed:
        if not valid_json:
            failure_category = "invalid_json"
        elif parsed is not None:
            action = parsed.get("action", "MISSING")
            if action not in ("tool_call", "final"):
                failure_category = "invalid_action"
            elif action == "tool_call":
                required = ["action", "tool_name", "arguments"]
                missing = [k for k in required if k not in parsed]
                if missing:
                    failure_category = "missing_required_top_level_fields"
                    missing_fields = missing
                else:
                    args = parsed.get("arguments", {})
                    fci_keys = [
                        "chef_name", "service_date", "start_time", "people",
                        "address", "cuisine", "budget_min", "budget_max",
                        "menu", "ingredient_purchase", "dietary_constraints", "occasion",
                    ]
                    missing_args = [k for k in fci_keys if k not in args]
                    if missing_args:
                        failure_category = "missing_find_chefs_argument_keys"
                        missing_fields = missing_args
                    else:
                        failure_category = "other_schema_error"
            elif action == "final":
                required = [
                    "action", "booking_state", "chef_query_status", "candidate_chefs",
                    "info_complete", "unrelated", "missing_info", "reply_type", "reply",
                ]
                missing = [k for k in required if k not in parsed]
                if missing:
                    failure_category = "missing_required_top_level_fields"
                    missing_fields = missing
                else:
                    rt = parsed.get("reply_type")
                    if rt not in CANONICAL_REPLY_TYPES:
                        failure_category = "invalid_reply_type"
                    else:
                        failure_category = "other_schema_error"

    return {
        "case_id": case.id, "label": label, "tags": case.tags,
        "raw_text": gen.raw_text,
        "finish_reason": gen.finish_reason,
        "completion_tokens": gen.completion_tokens,
        "parsed_json": parsed,
        "valid_json": valid_json,
        "protocol_pass": protocol.passed is True,
        "protocol_error": str(protocol.details.get("error", ""))[:200] if not protocol.passed else "",
        "task_correctness": task.score,
        "structured_score": task.details.get("structured_score", 0.0),
        "reply_score": task.details.get("reply_score", 0.0),
        "effective_pass": eff_pass,
        "failure_category": failure_category,
        "missing_fields": missing_fields,
        "action": parsed.get("action") if parsed else None,
        "reply_type": parsed.get("reply_type") if parsed else None,
        "tool_name": parsed.get("tool_name") if parsed else None,
        "arguments": parsed.get("arguments") if parsed else None,
        "latency_ms": gen.latency_ms,
        "ttft_ms": gen.ttft_ms,
        "throughput_source": gen.throughput_source,
    }


def run_experiment() -> dict:
    all_cases = load_eval_cases(Path("data/eval/frozen_test.jsonl"))
    case_map = {c.id: c for c in all_cases}
    cases = [case_map[cid] for cid in EXPERIMENT_CASE_IDS]

    output_dir = Path("reports/generated/phase02/structured_ab_10case")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_u = get_config_u()
    config_s = get_config_s()

    # ── CONTROL CHECK ─────────────────────────────────────
    assert config_u.use_structured_output is False
    assert config_s.use_structured_output is True
    assert config_u.model_id == config_s.model_id
    assert config_u.temperature == config_s.temperature
    assert config_u.max_new_tokens == config_s.max_new_tokens
    assert config_u.timeout_seconds == config_s.timeout_seconds

    for case in cases:
        u_msgs = build_prompt_u(case)
        s_msgs = build_prompt_u(case)
        assert u_msgs == s_msgs, f"Prompt mismatch for {case.id}"

    print("CONTROL_CHECK: PASS")
    print()

    # ── Run experiment ────────────────────────────────────
    results_u: list[dict] = []
    results_s: list[dict] = []

    for case in cases:
        print(f"  [U] {case.id}...", end=" ", flush=True)
        r_u = run_single_case(case, config_u, "U")
        results_u.append(r_u)
        print(f"pp={r_u['protocol_pass']} action={r_u['action']} fail={r_u['failure_category']}")

        print(f"  [S] {case.id}...", end=" ", flush=True)
        r_s = run_single_case(case, config_s, "S")
        results_s.append(r_s)
        print(f"pp={r_s['protocol_pass']} action={r_s['action']} fail={r_s['failure_category']}")

    # Save per-case details
    u_dir = output_dir / "U_unstructured"
    s_dir = output_dir / "S_structured"
    u_dir.mkdir(parents=True, exist_ok=True)
    s_dir.mkdir(parents=True, exist_ok=True)

    for r in results_u:
        (u_dir / f"{r['case_id']}_result.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for r in results_s:
        (s_dir / f"{r['case_id']}_result.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # Build summary
    summary = _build_summary(cases, results_u, results_s)
    (output_dir / "structured_ab_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(output_dir, summary)

    return summary


def _aggregate(results: list[dict]) -> dict:
    n = len(results)
    pp = sum(1 for r in results if r["protocol_pass"])
    vj = sum(1 for r in results if r["valid_json"])
    ep = sum(1 for r in results if r["effective_pass"])
    tc_mean = sum(r["task_correctness"] for r in results) / n if n else 0
    ss_mean = sum(r["structured_score"] for r in results) / n if n else 0
    rs_mean = sum(r["reply_score"] for r in results) / n if n else 0

    failures = Counter()
    for r in results:
        if not r["protocol_pass"]:
            failures[r["failure_category"]] += 1

    return {
        "total": n,
        "valid_json": vj, "protocol_pass": pp, "protocol_pass_rate": pp / n if n else 0,
        "effective_pass": ep, "effective_pass_rate": ep / n if n else 0,
        "mean_task_correctness": tc_mean,
        "mean_structured_score": ss_mean,
        "mean_reply_score": rs_mean,
        "failures": dict(failures),
    }


def _build_summary(
    cases: list[EvalCase], results_u: list[dict], results_s: list[dict]
) -> dict:
    sa = _aggregate(results_u)
    sb = _aggregate(results_s)

    delta = {}
    for key in ["valid_json", "protocol_pass", "effective_pass"]:
        delta[key] = sb[key] - sa[key]
    for key in ["mean_task_correctness", "mean_structured_score", "mean_reply_score"]:
        delta[key] = round(sb[key] - sa[key], 4)

    per_case = []
    for case, ru, rs in zip(cases, results_u, results_s, strict=True):
        per_case.append({
            "case_id": case.id,
            "tags": case.tags,
            "U_action": ru["action"], "S_action": rs["action"],
            "U_protocol_pass": ru["protocol_pass"], "S_protocol_pass": rs["protocol_pass"],
            "U_failure": ru["failure_category"], "S_failure": rs["failure_category"],
            "U_task_score": ru["task_correctness"], "S_task_score": rs["task_correctness"],
            "U_effective_pass": ru["effective_pass"], "S_effective_pass": rs["effective_pass"],
            "U_arguments": ru["arguments"], "S_arguments": rs["arguments"],
            "U_reply_type": ru["reply_type"], "S_reply_type": rs["reply_type"],
        })

    # Verdict
    u_pp = sa["protocol_pass"]
    s_pp = sb["protocol_pass"]
    s_tc = sb["mean_task_correctness"]
    s_ep = sb["effective_pass"]

    if u_pp <= 2 and s_pp >= 9 and (s_tc >= 0.8 or s_ep >= 8):
        verdict = (
            "STRONG EVIDENCE: The dominant failure is unconstrained schema serialization, "
            "not core business semantics. Runtime constrained decoding successfully separates "
            "deterministic schema compliance from model semantic reasoning."
        )
    elif s_pp >= 9 and 0.4 <= s_tc <= 0.7:
        verdict = (
            "MIXED: Schema serialization is a major failure source, but substantial "
            "business-semantic errors remain after structural constraints are removed."
        )
    elif s_pp >= 9 and s_tc <= 0.3:
        verdict = (
            "SEMANTIC BOTTLENECK REMAINS: Constrained decoding fixes structure, "
            "but the model still lacks sufficient HomeChef task capability."
        )
    elif s_pp < 8:
        verdict = (
            "INFRASTRUCTURE/SCHEMA MISMATCH: Structured decoding did not achieve "
            "expected Protocol pass rate. Check JSON Schema implementation, "
            "llama.cpp response_format compatibility, or schema generation."
        )
    else:
        verdict = "MIXED — results between boundary thresholds."

    return {
        "experiment": "Phase 02 Structured vs Unstructured",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "gguf": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "cases": EXPERIMENT_CASE_IDS,
        "U_summary": sa, "S_summary": sb,
        "delta": delta, "per_case": per_case,
        "verdict": verdict,
    }


def _write_markdown(output_dir: Path, summary: dict) -> None:
    sa, sb = summary["U_summary"], summary["S_summary"]
    delta = summary["delta"]

    lines = [
        "# Phase 02 Structured vs Unstructured Controlled Experiment",
        "",
        f"**Model:** {summary['model']}",
        f"**GGUF:** {summary['gguf']}",
        f"**Cases:** {len(summary['cases'])}",
        "",
        "## Summary",
        "",
        "| Metric | Unstructured | Structured | Delta |",
        "|--------|-------------|-----------|-------|",
        f"| Valid JSON | {sa['valid_json']}/{sa['total']} | {sb['valid_json']}/{sb['total']} | {delta['valid_json']:+d} |",
        f"| Protocol Pass | {sa['protocol_pass']}/{sa['total']} | {sb['protocol_pass']}/{sb['total']} | {delta['protocol_pass']:+d} |",
        f"| Effective Pass | {sa['effective_pass']}/{sa['total']} | {sb['effective_pass']}/{sb['total']} | {delta['effective_pass']:+d} |",
        f"| Mean Task Correctness | {sa['mean_task_correctness']:.4f} | {sb['mean_task_correctness']:.4f} | {delta['mean_task_correctness']:+.4f} |",
        f"| Mean Structured Score | {sa['mean_structured_score']:.4f} | {sb['mean_structured_score']:.4f} | {delta['mean_structured_score']:+.4f} |",
        f"| Mean Reply Score | {sa['mean_reply_score']:.4f} | {sb['mean_reply_score']:.4f} | {delta['mean_reply_score']:+.4f} |",
        "",
        "### U (Unstructured) Failures",
    ]
    for cat, cnt in sorted(sa["failures"].items()):
        lines.append(f"- {cat}: {cnt}")
    lines.append("")
    lines.append("### S (Structured) Failures")
    for cat, cnt in sorted(sb["failures"].items()):
        lines.append(f"- {cat}: {cnt}")

    lines.extend([
        "",
        "## Per-Case",
        "",
        "| Case ID | Tags | U PP | S PP | U Failure | S Failure | U Action | S Action | U Task | S Task |",
        "|---------|------|------|------|-----------|-----------|----------|----------|--------|--------|",
    ])
    for pc in summary["per_case"]:
        tags_str = ",".join(pc["tags"][:2])
        lines.append(
            f"| {pc['case_id']} | {tags_str} | {pc['U_protocol_pass']} | {pc['S_protocol_pass']} | "
            f"{pc['U_failure']} | {pc['S_failure']} | "
            f"{pc['U_action']} | {pc['S_action']} | "
            f"{pc['U_task_score']:.3f} | {pc['S_task_score']:.3f} |"
        )

    lines.extend([
        "",
        "## Verdict",
        "",
        summary["verdict"],
    ])

    (output_dir / "structured_ab_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 02 Structured vs Unstructured Experiment")
    print("10 cases x 2 modes = 20 generations")
    print("=" * 60)
    print()

    result = run_experiment()

    print()
    print("=" * 60)
    print(f"U: Protocol Pass = {result['U_summary']['protocol_pass']}/{result['U_summary']['total']}")
    print(f"S: Protocol Pass = {result['S_summary']['protocol_pass']}/{result['S_summary']['total']}")
    print(f"Delta: {result['delta']['protocol_pass']:+d}")
    print(f"Verdict: {result['verdict']}")
    print("=" * 60)
