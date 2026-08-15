"""Phase 02 Prompt A/B Controlled Experiment: 10 cases, 4B-Instruct-2507.

A = Current production PromptBuilder output.
B = Same PromptBuilder + STRICT OUTPUT REQUIREMENTS block appended.

Verifies whether Prompt contract explicitness alone improves Protocol Pass Rate
without constrained decoding.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

from homechef_booking.evaluation.evidence import derive_tool_evidence
from homechef_booking.evaluation.sample import EvalCase, load_eval_cases
from homechef_booking.evaluation.scorers.protocol import ProtocolScorer
from homechef_booking.evaluation.scorers.task_correctness import TaskCorrectnessScorer
from homechef_booking.inference.backend import GenerationParams
from homechef_booking.inference.factory import load_backend
from homechef_booking.inference.llama_cpp_backend import LlamaCppServerConfig
from homechef_booking.inference.runner import run_inference
from homechef_booking.prompts.template import PromptBuilder
from homechef_booking.schemas.booking import ReplyType

# ── Canonical ReplyType enum (read from code, not hardcoded) ──
CANONICAL_REPLY_TYPES = sorted(e.value for e in ReplyType)

# ── STRICT OUTPUT REQUIREMENTS block (appended to B's system message) ──
STRICT_OUTPUT_BLOCK = f"""
STRICT OUTPUT REQUIREMENTS:

1. The value of "action" MUST be exactly one of the action values defined by the output contract.  The only valid values are "tool_call" and "final".

2. For a tool call:
   - "action" MUST be exactly the literal string "tool_call".
   - "tool_name" MUST be exactly the literal string "find_chefs".
   - NEVER use "find_chefs" as the value of "action".  The value of "action" is always either "tool_call" or "final".

3. Every key listed in "tool_call_required_keys" MUST appear in every tool-call output.  No required key may be omitted.

4. Every key listed in "final_required_keys" MUST appear in every final output.  No required key may be omitted.

5. Every key listed in "find_chefs_argument_keys" MUST appear inside "arguments".  Even when a value is unknown or empty, the key MUST still appear using the contract-appropriate null, false, or [] value.

6. A field MUST NOT be omitted merely because its value appears obvious, redundant, empty, null, false, or already exists in the conversation state.

7. "reply_type" MUST be one of these canonical ReplyType values: {', '.join(repr(v) for v in CANONICAL_REPLY_TYPES)}.  Do NOT invent a new reply_type value.

8. Do NOT invent new action values, field names, or enum values.

9. Output exactly one JSON object and nothing else.
"""

# ── Inference config for 4B-Instruct-2507 ──
EXPERIMENT_CONFIG = LlamaCppServerConfig(
    base_url="http://127.0.0.1:8080/v1",
    model_id="Qwen/Qwen3-4B-Instruct-2507",
    max_new_tokens=512,
    temperature=0.0,
    timeout_seconds=900,
    use_structured_output=False,
    device="cpu",
    runtime="llama.cpp",
    model_format="gguf",
    quantization="Q4_K_M",
    gpu_layers=0,
)


def select_10_cases() -> list[EvalCase]:
    """Select 2 cases from each of 5 categories, in Frozen Test order."""
    all_cases = load_eval_cases(Path("data/eval/frozen_test.jsonl"))

    categories: dict[str, list[EvalCase]] = {
        "missing_required_slots": [],
        "valid_search_tool_call": [],
        "tool_result": [],
        "relative_time": [],
        "semantic_slots": [],
    }

    for case in all_cases:
        tags = set(case.tags)
        if "missing_required_slots" in tags and len(categories["missing_required_slots"]) < 2:
            categories["missing_required_slots"].append(case)
        if "valid_search_tool_call" in tags and len(categories["valid_search_tool_call"]) < 2:
            categories["valid_search_tool_call"].append(case)
        if "tool_result" in tags and len(categories["tool_result"]) < 2:
            categories["tool_result"].append(case)
        if "relative_time" in tags and len(categories["relative_time"]) < 2:
            categories["relative_time"].append(case)
        if "semantic_slots" in tags and len(categories["semantic_slots"]) < 2:
            categories["semantic_slots"].append(case)

        if all(len(v) >= 2 for v in categories.values()):
            break

    # Flatten in original Frozen Test order
    selected: list[EvalCase] = []
    all_ids = [c.id for c in all_cases]
    selected_ids = set()
    for cat_cases in categories.values():
        for case in cat_cases:
            if case.id not in selected_ids:
                selected.append(case)
                selected_ids.add(case.id)

    selected.sort(key=lambda c: all_ids.index(c.id))
    return selected


def build_prompt_a(case: EvalCase) -> list[dict]:
    """Build messages using production PromptBuilder (unchanged)."""
    builder = PromptBuilder()
    return builder.build_messages(case.input)


def build_prompt_b(case: EvalCase) -> list[dict]:
    """Build messages = Prompt A + STRICT_OUTPUT_BLOCK appended to system msg."""
    messages_a = build_prompt_a(case)
    messages_b = deepcopy(messages_a)

    for msg in messages_b:
        if msg["role"] == "system":
            msg["content"] = msg["content"] + STRICT_OUTPUT_BLOCK
            break

    return messages_b


def get_experiment_config() -> LlamaCppServerConfig:
    return EXPERIMENT_CONFIG


def run_single_case(
    case: EvalCase, messages: list[dict], label: str
) -> dict:
    """Run one case through the backend and score it.

    Returns dict with full results.
    """
    # Build backend and generate
    backend = load_backend(Path("configs/inference/llama_cpp_4b_instruct_2507_cpu.yaml"))
    params = GenerationParams()

    gen = run_inference(case.id, case.input, backend, params)

    # Score
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

    # Parse raw JSON for failure taxonomy
    raw = gen.raw_text or ""
    json_str = re.sub(r'^```(?:json)?\s*\n?', '', raw.strip())
    json_str = re.sub(r'\n?```\s*$', '', json_str)
    try:
        parsed = json.loads(json_str)
        valid_json = True
    except json.JSONDecodeError:
        parsed = None
        valid_json = False

    # Failure classification
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
                    # Check FindChefsInput keys
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
        "case_id": case.id,
        "label": label,
        "tags": case.tags,
        "raw_text": gen.raw_text,
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
        "latency_ms": gen.latency_ms,
        "ttft_ms": gen.ttft_ms,
        "completion_tokens": gen.completion_tokens,
    }


def run_experiment() -> dict:
    """Run full A/B experiment: 10 cases x 2 prompts."""
    cases = select_10_cases()
    output_dir = Path("reports/generated/phase02/prompt_ab_10case")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save selected cases
    selected_info = []
    all_cases = load_eval_cases(Path("data/eval/frozen_test.jsonl"))
    all_ids = [c.id for c in all_cases]
    for case in cases:
        selected_info.append({
            "case_id": case.id,
            "original_index": all_ids.index(case.id),
            "tags": case.tags,
            "output_kind": case.output_kind,
        })
    (output_dir / "selected_cases.json").write_text(
        json.dumps(selected_info, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save 10-case JSONL
    frozen_10_path = output_dir / "frozen_10cases.jsonl"
    with open(Path("data/eval/frozen_test.jsonl"), encoding="utf-8") as src:
        all_lines = src.readlines()
    selected_indices = {info["original_index"] for info in selected_info}
    with open(frozen_10_path, "w", encoding="utf-8") as dst:
        for i, line in enumerate(all_lines):
            if i in selected_indices:
                dst.write(line)

    # CONTROL_CHECK before running
    for case in cases:
        msgs_a = build_prompt_a(case)
        msgs_b = build_prompt_b(case)
        assert len(msgs_a) == len(msgs_b), f"Message count mismatch for {case.id}"
        for i, (ma, mb) in enumerate(zip(msgs_a, msgs_b, strict=True)):
            if ma["role"] != "system":
                assert ma == mb, f"Non-system msg {i} differs for {case.id}"
        sys_a = next(m for m in msgs_a if m["role"] == "system")
        sys_b = next(m for m in msgs_b if m["role"] == "system")
        assert sys_a["content"] in sys_b["content"], f"B missing A content for {case.id}"
        assert STRICT_OUTPUT_BLOCK.strip() in sys_b["content"], f"B missing strict block for {case.id}"

    # Run A then B, interleaved per case
    results_a: list[dict] = []
    results_b: list[dict] = []

    for case in cases:
        # Save A messages
        msgs_a = build_prompt_a(case)
        a_dir = output_dir / "A_current_prompt"
        a_dir.mkdir(parents=True, exist_ok=True)
        (a_dir / f"{case.id}_messages.json").write_text(
            json.dumps(msgs_a, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"  [A] {case.id}...", end=" ", flush=True)
        result_a = run_single_case(case, msgs_a, "A")
        results_a.append(result_a)
        print(f"pp={result_a['protocol_pass']} action={result_a['action']}")

        # Save A raw output
        (a_dir / f"{case.id}_raw_output.txt").write_text(
            result_a["raw_text"] or "", encoding="utf-8")

        # Save B messages
        msgs_b = build_prompt_b(case)
        b_dir = output_dir / "B_explicit_contract"
        b_dir.mkdir(parents=True, exist_ok=True)
        (b_dir / f"{case.id}_messages.json").write_text(
            json.dumps(msgs_b, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"  [B] {case.id}...", end=" ", flush=True)
        result_b = run_single_case(case, msgs_b, "B")
        results_b.append(result_b)
        print(f"pp={result_b['protocol_pass']} action={result_b['action']}")

        # Save B raw output
        (b_dir / f"{case.id}_raw_output.txt").write_text(
            result_b["raw_text"] or "", encoding="utf-8")

    # Aggregate results
    summary = _build_summary(cases, results_a, results_b)
    (output_dir / "prompt_ab_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_markdown(output_dir, summary)

    return summary


def _build_summary(
    cases: list[EvalCase],
    results_a: list[dict],
    results_b: list[dict],
) -> dict:
    """Build aggregate A/B summary."""

    def aggregate(results: list[dict]) -> dict:
        n = len(results)
        pp = sum(1 for r in results if r["protocol_pass"])
        vj = sum(1 for r in results if r["valid_json"])
        ep = sum(1 for r in results if r["effective_pass"])
        tc_mean = sum(r["task_correctness"] for r in results) / n if n else 0

        failures = Counter()
        for r in results:
            if not r["protocol_pass"]:
                failures[r["failure_category"]] += 1

        return {
            "total": n,
            "valid_json": vj,
            "protocol_pass": pp,
            "protocol_pass_rate": pp / n if n else 0,
            "effective_pass": ep,
            "effective_pass_rate": ep / n if n else 0,
            "mean_task_correctness": tc_mean,
            "failures": dict(failures),
        }

    summary_a = aggregate(results_a)
    summary_b = aggregate(results_b)

    delta = {}
    for key in ["valid_json", "protocol_pass", "effective_pass"]:
        delta[key] = summary_b[key] - summary_a[key]
    delta["mean_task_correctness"] = round(
        summary_b["mean_task_correctness"] - summary_a["mean_task_correctness"], 4)

    # Per-case table
    per_case = []
    for case, ra, rb in zip(cases, results_a, results_b, strict=True):
        per_case.append({
            "case_id": case.id,
            "tags": case.tags,
            "A_protocol_pass": ra["protocol_pass"],
            "B_protocol_pass": rb["protocol_pass"],
            "A_failure": ra["failure_category"],
            "B_failure": rb["failure_category"],
            "A_action": ra["action"],
            "B_action": rb["action"],
            "A_missing_fields": ra["missing_fields"],
            "B_missing_fields": rb["missing_fields"],
            "A_task_score": ra["task_correctness"],
            "B_task_score": rb["task_correctness"],
        })

    # Verdict
    a_pp = summary_a["protocol_pass"]
    b_pp = summary_b["protocol_pass"]
    if b_pp >= 8 and a_pp <= 3:
        verdict = "Strong evidence that Prompt contract explicitness is a major cause of the 4B unconstrained protocol failure."
    elif b_pp - a_pp >= 4 and b_pp < 8:
        verdict = "Prompt under-specification is a material contributor, but natural-language prompting alone is insufficient for reliable strict-schema conformance."
    elif b_pp - a_pp <= 2:
        verdict = "Prompt explicitness is not the primary explanation; investigate model/instruction-tuning behavior or use constrained decoding."
    else:
        verdict = "MIXED — results are between boundary thresholds."

    return {
        "experiment": "Phase 02 Prompt A/B Controlled",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "gguf": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "cases": [c.id for c in cases],
        "num_cases": len(cases),
        "A_summary": summary_a,
        "B_summary": summary_b,
        "delta": delta,
        "per_case": per_case,
        "verdict": verdict,
    }


def _write_markdown(output_dir: Path, summary: dict) -> None:
    """Generate prompt_ab_summary.md."""
    sa = summary["A_summary"]
    sb = summary["B_summary"]
    delta = summary["delta"]

    lines = [
        "# Phase 02 Prompt A/B Controlled Experiment",
        "",
        f"**Model:** {summary['model']}",
        f"**GGUF:** {summary['gguf']}",
        f"**Cases:** {summary['num_cases']}",
        "",
        "## Summary",
        "",
        "| Metric | A (Current) | B (Explicit) | Delta |",
        "|--------|------------|-------------|-------|",
        f"| Valid JSON | {sa['valid_json']}/{sa['total']} | {sb['valid_json']}/{sb['total']} | {delta['valid_json']:+d} |",
        f"| Protocol Pass | {sa['protocol_pass']}/{sa['total']} | {sb['protocol_pass']}/{sb['total']} | {delta['protocol_pass']:+d} |",
        f"| Effective Pass | {sa['effective_pass']}/{sa['total']} | {sb['effective_pass']}/{sb['total']} | {delta['effective_pass']:+d} |",
        f"| Mean Task Correctness | {sa['mean_task_correctness']:.4f} | {sb['mean_task_correctness']:.4f} | {delta['mean_task_correctness']:+.4f} |",
        "",
        "### A Failures",
    ]
    for cat, count in sorted(sa["failures"].items()):
        lines.append(f"- {cat}: {count}")
    lines.append("")
    lines.append("### B Failures")
    for cat, count in sorted(sb["failures"].items()):
        lines.append(f"- {cat}: {count}")

    lines.extend([
        "",
        "## Per-Case",
        "",
        "| Case ID | Tags | A PP | B PP | A Failure | B Failure | A Action | B Action | A Missing | B Missing | A Task | B Task |",
        "|---------|------|------|------|-----------|-----------|----------|----------|-----------|-----------|--------|--------|",
    ])
    for pc in summary["per_case"]:
        tags_str = ",".join(pc["tags"][:2])
        lines.append(
            f"| {pc['case_id']} | {tags_str} | {pc['A_protocol_pass']} | {pc['B_protocol_pass']} | "
            f"{pc['A_failure']} | {pc['B_failure']} | "
            f"{pc['A_action']} | {pc['B_action']} | "
            f"{pc['A_missing_fields']} | {pc['B_missing_fields']} | "
            f"{pc['A_task_score']:.3f} | {pc['B_task_score']:.3f} |"
        )

    lines.extend([
        "",
        "## Verdict",
        "",
        summary["verdict"],
    ])

    (output_dir / "prompt_ab_summary.md").write_text(
        "\n".join(lines), encoding="utf-8")


# ── CLI entry point ─────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 02 Prompt A/B Controlled Experiment")
    print("Model: Qwen3-4B-Instruct-2507 Q4_K_M")
    print("10 cases x 2 prompts = 20 generations")
    print("=" * 60)
    print()

    result = run_experiment()

    print()
    print("=" * 60)
    print(f"A: Protocol Pass = {result['A_summary']['protocol_pass']}/{result['A_summary']['total']}")
    print(f"B: Protocol Pass = {result['B_summary']['protocol_pass']}/{result['B_summary']['total']}")
    print(f"Delta: {result['delta']['protocol_pass']:+d}")
    print(f"Verdict: {result['verdict']}")
    print("=" * 60)
