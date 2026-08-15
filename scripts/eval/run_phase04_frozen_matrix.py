"""Phase 04 Frozen Test evaluation — 12-cell exploratory matrix runner.

Runs the 12-cell exploratory_currentdata evaluation:

    6 checkpoints x 2 runtime modes (U/S) = 12 cells
    1.7B/4B x {SFT, DPO-β.1, DPO-β.3} x {U, S}

Each cell loads the base model + LoRA adapter via the existing
``phase04_hf_backend`` (reused, NOT a new inference protocol), runs the SAME
Frozen Test (120 cases, fixed SHA256), scores via the Phase02 Frozen Benchmark
contract, writes per-cell artifacts, then releases model/tokenizer/CUDA cache.

Fail-closed:
- missing adapter -> hard fail (no Base fallback).
- structured constraint init failure -> hard fail (no U fallback).

Supports --skip-complete so an interrupted run can resume.

Usage:
    python scripts/eval/run_phase04_frozen_matrix.py \
        --output-root reports/generated/phase04/exploratory_currentdata \
        --skip-complete
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from homechef_booking.evaluation.evidence import derive_tool_evidence
from homechef_booking.evaluation.sample import load_eval_cases
from homechef_booking.evaluation.scorers.protocol import ProtocolScorer
from homechef_booking.evaluation.scorers.task_correctness import TaskCorrectnessScorer
from homechef_booking.inference.backend import GenerationParams
from homechef_booking.inference.phase04_hf_backend import (
    Phase04HFConfig,
    Phase04HFTransformersBackend,
)
from homechef_booking.inference.runner import run_inference
from homechef_booking.training.formal_matrix import (
    FROZEN_TEST_PATH,
    FROZEN_TEST_SHA256,
    FROZEN_TEST_TOTAL_CASES,
    KNOWN_DATA_LIMITATION,
    build_phase04_exploratory_matrix,
    validate_exploratory_matrix,
)

EXPLORATORY_CLASS = "exploratory_currentdata"


# ── Scoring (reuse Phase02 Frozen Benchmark contract) ────────────────────────
def _score_case(case, gen) -> dict:
    protocol_scorer = ProtocolScorer()
    task_scorer = TaskCorrectnessScorer()
    protocol = protocol_scorer.score(case, gen)

    parseable = False
    pred = {}
    try:
        pred = json.loads(gen.raw_text or "")
        parseable = isinstance(pred, dict)
    except Exception:
        pred = {}

    evidence = None
    try:
        evidence = derive_tool_evidence(case.input)
    except Exception:
        evidence = None

    task = task_scorer.score(case, gen, evidence if parseable else None)
    eff = protocol.passed is True and task.score >= 0.95

    return {
        "case_id": case.id,
        "tags": case.tags,
        "output_kind": case.output_kind,
        "raw_text": gen.raw_text,
        "parsed_json": pred if parseable else None,
        "valid_json": parseable,
        "protocol_pass": protocol.passed is True,
        "protocol_error": str(protocol.details.get("error", ""))[:200] if not protocol.passed else "",
        "task_correctness": task.score,
        "structured_score": task.details.get("structured_score", 0.0),
        "reply_score": task.details.get("reply_score", 0.0),
        "effective_pass": eff,
        "action": pred.get("action") if parseable else None,
        "reply_type": pred.get("reply_type") if parseable else None,
        "latency_ms": gen.latency_ms,
        "ttft_ms": gen.ttft_ms,
        "completion_tokens": gen.completion_tokens,
        "tokens_per_second": gen.tokens_per_second,
        "throughput_source": gen.throughput_source,
        "finish_reason": gen.finish_reason,
        # FIX 4 — generation provenance; null on success (fail-closed guarantees).
        "generation_error_type": gen.error_type,
        "generation_error_message": gen.error_message,
    }


def _aggregate(results: list[dict]) -> dict:
    """Aggregate case results into a scorecard (Phase02 canonical metrics)."""
    n = len(results)
    pp = sum(1 for r in results if r["protocol_pass"])
    ep = sum(1 for r in results if r["effective_pass"])
    vj = sum(1 for r in results if r["valid_json"])
    tc = statistics.mean([r["task_correctness"] for r in results])
    ss = statistics.mean([r["structured_score"] for r in results])
    rs = statistics.mean([r["reply_score"] for r in results])

    fails = Counter(r.get("primary_failure") or "unknown" for r in results if not r["protocol_pass"])
    lats = [r["latency_ms"] for r in results if r["latency_ms"] and r["latency_ms"] > 0]
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] and r["ttft_ms"] > 0]
    tpses = [r["tokens_per_second"] for r in results if r["tokens_per_second"] and r["tokens_per_second"] > 0]

    def _p95(vals):
        if not vals:
            return None
        k = 0.95 * (len(vals) - 1)
        f = int(k)
        c = k - f
        return vals[f] + c * (vals[f + 1] - vals[f]) if f + 1 < len(vals) else vals[f]

    # FIX 3 — generation validity counts. With generation fail-closed these are
    # guaranteed to be success=n, failure=0, raw_text_non_null=n for a completed
    # cell; the counts are persisted for independent verification.
    generation_success_count = sum(
        1 for r in results
        if r.get("generation_error_type") is None
        and r.get("raw_text") not in (None, "")
    )
    generation_failure_count = sum(
        1 for r in results if r.get("generation_error_type") is not None
    )
    raw_text_non_null_count = sum(
        1 for r in results if r.get("raw_text") is not None
    )

    return {
        "protocol_pass_rate": round(pp / n, 4),
        "effective_pass_rate": round(ep / n, 4),
        "valid_json_rate": round(vj / n, 4),
        "mean_task_correctness": round(tc, 4),
        "mean_structured_score": round(ss, 4),
        "mean_reply_score": round(rs, 4),
        "critical_error_rate": round(sum(1 for r in results if not r["protocol_pass"]) / n, 4),
        "primary_failures": dict(fails),
        "total_cases": n,
        "generation_success_count": generation_success_count,
        "generation_failure_count": generation_failure_count,
        "raw_text_non_null_count": raw_text_non_null_count,
        "performance": {
            "p95_latency_s": round(_p95(sorted(lats)) / 1000, 3) if lats else None,
            "mean_ttft_s": round(statistics.mean(ttfts) / 1000, 3) if ttfts else None,
            "mean_tokens_per_second": round(statistics.mean(tpses), 2) if tpses else None,
            "latency_sample_count": len(lats),
        },
    }


# ── Per-cell artifacts ───────────────────────────────────────────────────────
def _run_manifest(run, frozen_sha: str, total: int) -> dict:
    return {
        "run_id": run.run_id,
        "experiment_class": EXPLORATORY_CLASS,
        "model_id": run.model_id,
        "stage": run.stage,
        "beta": run.pref_beta,
        "variant": run.variant,
        "use_structured_output": run.use_structured_output,
        "adapter_name_or_path": str(run.adapter_name_or_path),
        "frozen_test_path": str(FROZEN_TEST_PATH),
        "frozen_test_sha256": frozen_sha,
        "total_cases": total,
        "generation_success_count": None,  # populated by _aggregate before write
        "generation_failure_count": None,
        "raw_text_non_null_count": None,
        "known_data_limitation": KNOWN_DATA_LIMITATION,
        "formal_release_eligible": False,
    }


def _is_complete(run_dir: Path) -> bool:
    """A cell is complete if all artifacts exist AND generation validity holds:
    total_cases==120, generation_success_count==120, generation_failure_count==0,
    raw_text_non_null_count==120."""
    if not (run_dir / "scorecard.json").exists():
        return False
    if not (run_dir / "case_results.json").exists():
        return False
    if not (run_dir / "run_manifest.json").exists():
        return False
    try:
        sc = json.loads((run_dir / "scorecard.json").read_text(encoding="utf-8"))
        return (
            sc.get("total_cases") == FROZEN_TEST_TOTAL_CASES
            and sc.get("generation_success_count") == FROZEN_TEST_TOTAL_CASES
            and sc.get("generation_failure_count") == 0
            and sc.get("raw_text_non_null_count") == FROZEN_TEST_TOTAL_CASES
        )
    except Exception:
        return False


def _failure_analysis(results: list[dict]) -> dict:
    eff_fail = [r for r in results if r["protocol_pass"] and r["task_correctness"] < 0.95]
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
        "protocol_fail_count": sum(1 for r in results if not r["protocol_pass"]),
        "task_score_buckets": buckets,
        "primary_failures": dict(Counter(r.get("primary_failure") or "unknown"
                                         for r in results if not r["protocol_pass"])),
    }


def _run_cell(run, cases, frozen_sha: str) -> None:
    """Execute one cell: load base+adapter, run Frozen Test, score, write artifacts."""
    run_dir = run.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Fail-closed: adapter must exist (no Base fallback).
    adapter = run.adapter_name_or_path
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Adapter not found for {run.run_id}: {adapter} (no adapter_config.json). "
            "Cannot evaluate; no Base fallback."
        )

    config = Phase04HFConfig(
        model_id=run.model_id,
        adapter_name_or_path=str(adapter),
        device="auto",
        torch_dtype="bfloat16",
        max_new_tokens=512,
        use_structured_output=run.use_structured_output,
        model_key=run.model_key,
        training_stage=run.stage,
        model_size=run.model_size,
        pref_beta=run.pref_beta,
    )
    backend = Phase04HFTransformersBackend()
    backend.load(config)  # raises hard-fail if adapter/structured init fails

    params = GenerationParams()
    results = []
    try:
        for case in cases:
            # Canonical inference path (FIX 1): reuse run_inference so the
            # PromptBuilder builds list[Message] for backend.generate. Never pass
            # BookingRuntimeInput directly to the backend.
            gen = run_inference(case.id, case.input, backend, params)
            # Generation fail-closed (FIX 2): any generation error OR a
            # None/empty raw_text is a hard cell failure — no scorer, no scorecard.
            if gen.error_type is not None:
                raise RuntimeError(
                    f"[{run.run_id}] generation error for case {case.id}: "
                    f"error_type={gen.error_type!r} error_message={gen.error_message!r}"
                )
            if gen.raw_text is None or gen.raw_text.strip() == "":
                raise RuntimeError(
                    f"[{run.run_id}] empty raw_text for case {case.id}: "
                    f"raw_text={gen.raw_text!r}"
                )
            results.append(_score_case(case, gen))
    finally:
        # Release model/tokenizer/CUDA cache per cell.
        backend._model = None
        backend._tokenizer = None
        if hasattr(backend, "_constraint_fn"):
            backend._constraint_fn = None
        try:
            import gc

            import torch

            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

    scorecard = _aggregate(results)
    manifest = _run_manifest(run, frozen_sha, len(results))
    # FIX 3 — populate generation validity counts from the actual scorecard so
    # the manifest independently proves all 120 generations succeeded.
    manifest["generation_success_count"] = scorecard["generation_success_count"]
    manifest["generation_failure_count"] = scorecard["generation_failure_count"]
    manifest["raw_text_non_null_count"] = scorecard["raw_text_non_null_count"]
    (run_dir / "scorecard.json").write_text(
        json.dumps(scorecard, indent=2), encoding="utf-8")
    (run_dir / "case_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "failure_analysis.json").write_text(
        json.dumps(_failure_analysis(results), indent=2), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  {run.run_id}: protocol={scorecard['protocol_pass_rate']} "
          f"task={scorecard['mean_task_correctness']} eff={scorecard['effective_pass_rate']} "
          f"gen_success={scorecard['generation_success_count']}/{scorecard['total_cases']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase04 12-cell Frozen Test evaluation")
    parser.add_argument("--output-root", type=str,
                        default="reports/generated/phase04/exploratory_currentdata")
    parser.add_argument("--skip-complete", action="store_true",
                        help="Skip cells that already have complete artifacts")
    args = parser.parse_args()

    # Frozen Test SHA gate.
    frozen_sha = hashlib.sha256(FROZEN_TEST_PATH.read_bytes()).hexdigest()
    if frozen_sha != FROZEN_TEST_SHA256:
        print(f"FAIL: Frozen Test SHA mismatch. Got {frozen_sha}, expected {FROZEN_TEST_SHA256}")
        raise SystemExit(1)

    cases = load_eval_cases(FROZEN_TEST_PATH)
    if len(cases) != FROZEN_TEST_TOTAL_CASES:
        print(f"FAIL: Frozen Test has {len(cases)} cases, expected {FROZEN_TEST_TOTAL_CASES}")
        raise SystemExit(1)

    runs = build_phase04_exploratory_matrix(Path(args.output_root))
    errors = validate_exploratory_matrix(runs)
    if errors:
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print(f"Phase04 12-cell Frozen matrix OK: {len(runs)} cells")
    for run in runs:
        if args.skip_complete and _is_complete(run.output_dir):
            print(f"  {run.run_id}: SKIP (complete)")
            continue
        _run_cell(run, cases, frozen_sha)


if __name__ == "__main__":
    main()
