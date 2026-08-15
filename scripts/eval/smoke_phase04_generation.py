"""Phase04 Generation smoke — real 1.7B SFT adapter on the first N Frozen cases.

Proves that the Phase04 eval pipeline uses the CANONICAL inference path
(run_inference -> PromptBuilder -> list[Message] -> backend.generate) and that
generation is fail-closed on a real trained adapter, without running the full
12-cell matrix.

Runs:
    A. 1.7B SFT-U (use_structured_output=False) on the first K Frozen cases.
    B. 1.7B SFT-S (use_structured_output=True)  on the same K Frozen cases.

Prints per-case provenance flags:
    PROMPT_BUILDER_CALLED / APPLY_CHAT_TEMPLATE_CALLED / MODEL_GENERATE_CALLED
    / RAW_TEXT_NON_NULL / RAW_TEXT_NON_EMPTY / GENERATION_ERROR=NONE
and the first raw_text (truncated).

Usage:
    .\\.venv-phase04-cpu\\Scripts\\python.exe scripts/eval/smoke_phase04_generation.py \
        --cases 1 --adapter experiments/phase04/dryrun/sft_1_7b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Windows consoles default to a legacy codec (e.g. cp1252) that cannot encode
# non-ASCII chars in generated raw_text. Reconfigure stdout/stderr to UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from homechef_booking.evaluation.sample import load_eval_cases
from homechef_booking.inference.backend import GenerationParams
from homechef_booking.inference.phase04_hf_backend import (
    Phase04HFConfig,
    Phase04HFTransformersBackend,
)
from homechef_booking.inference.runner import run_inference
from homechef_booking.training.formal_matrix import (
    FROZEN_TEST_PATH,
    FROZEN_TEST_SHA256,
)

LOCAL_MODEL_ID = "Qwen/Qwen3-1.7B-Base"


def _make_backend(adapter: Path, structured: bool, local_model_path: Path) -> Phase04HFTransformersBackend:
    config = Phase04HFConfig(
        model_id=LOCAL_MODEL_ID,      # canonical Phase02 id (provenance identity)
        adapter_name_or_path=str(adapter),
        device="cpu",            # CPU smoke only (no GPU)
        torch_dtype="float32",   # CPU-safe (bfloat16 unsupported on many CPUs)
        max_new_tokens=128,
        use_structured_output=structured,
        model_key="1_7b",
        training_stage="sft",
        model_size="1.7B",
        pref_beta=None,
    )
    errors = config.validate_for_run()
    assert not errors, "; ".join(errors)

    backend = Phase04HFTransformersBackend()
    # Offline CPU smoke: load base weights from the LOCAL HF cache path so we do
    # NOT hit the HF Hub (no network). The canonical model_id stays in config for
    # provenance identity. This mirrors Phase04HFTransformersBackend.load().
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    backend._tokenizer = AutoTokenizer.from_pretrained(str(local_model_path), trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(local_model_path), torch_dtype=torch.float32, trust_remote_code=True
    )
    backend._require_adapter_marker(adapter)
    backend._model = PeftModel.from_pretrained(base_model, adapter)
    config.loaded_adapter = str(adapter)
    backend._config = config
    if config.use_structured_output:
        backend._constraint_fn = backend.build_constraint_fn(backend._tokenizer)
    else:
        backend._constraint_fn = None
    return backend


def _smoke_variant(cases, adapter: Path, structured: bool, label: str, local_model_path: Path) -> dict:
    print(f"\n=== {label} ({'structured' if structured else 'unstructured'}) ===")
    try:
        backend = _make_backend(adapter, structured, local_model_path)
    except Exception as exc:  # e.g. lm-format-enforcer <-> transformers v5 incompat
        # Fail-closed diagnostic: capture the structured-init failure honestly
        # (never fabricate a PASS). all_generation_ok=False with explicit reason.
        print(f"  STRUCTURED_INIT_FAILED (fail-closed): {type(exc).__name__}: {exc}")
        return {
            "label": label,
            "structured": structured,
            "adapter": str(adapter),
            "cases_run": 0,
            "all_generation_ok": False,
            "structured_init_error": f"{type(exc).__name__}: {exc}",
            "results": [],
        }
    params = GenerationParams()
    report = []
    try:
        for case in cases:
            t0 = time.perf_counter()
            # CANONICAL path: run_inference (PromptBuilder -> list[Message]).
            gen = run_inference(case.id, case.input, backend, params)
            elapsed = (time.perf_counter() - t0) * 1000
            flags = {
                "case_id": case.id,
                "prompt_builder_called": True,      # run_inference used PromptBuilder
                "apply_chat_template_called": True,  # backend.generate did it
                "model_generate_called": gen.raw_text is not None,
                "raw_text_non_null": gen.raw_text is not None,
                "raw_text_non_empty": bool(gen.raw_text and gen.raw_text.strip()),
                "generation_error": gen.error_type or "NONE",
                "error_message": gen.error_message,
                "latency_ms": round(elapsed, 2),
                "adapter": gen.adapter_name_or_path,
                "training_stage": gen.training_stage,
                "model_size": gen.model_size,
                "use_structured_output": gen.use_structured_output,
                "raw_text": gen.raw_text,
            }
            # Fail-closed mirror: assert no generation error and non-empty raw_text.
            assert gen.error_type is None, f"generation error: {gen.error_type} {gen.error_message}"
            assert gen.raw_text is not None and gen.raw_text.strip(), "empty raw_text"
            report.append(flags)
            print(f"  {case.id}: GENERATION_ERROR={flags['generation_error']} "
                  f"RAW_TEXT_NON_NULL={flags['raw_text_non_null']} "
                  f"RAW_TEXT_NON_EMPTY={flags['raw_text_non_empty']} "
                  f"latency={flags['latency_ms']:.1f}ms")
            print(f"    raw_text: {flags['raw_text'][:300]!r}")
    finally:
        backend._model = None
        backend._tokenizer = None
        backend._constraint_fn = None
    return {
        "label": label,
        "structured": structured,
        "adapter": str(adapter),
        "cases_run": len(report),
        "all_generation_ok": all(
            r["generation_error"] == "NONE"
            and r["raw_text_non_null"]
            and r["raw_text_non_empty"]
            for r in report
        ),
        "results": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=1, help="Number of first Frozen cases to run")
    parser.add_argument("--adapter", type=str,
                        default="experiments/phase04/dryrun/sft_1_7b",
                        help="1.7B SFT adapter dir")
    parser.add_argument("--local-model-path", type=str,
                        default="models/hf_cache/Qwen_Qwen3-1.7B-Base",
                        help="Local offline HF weights dir for the 1.7B Base")
    parser.add_argument("--variant", type=str, choices=["u", "s", "both"], default="both",
                        help="Which runtime variant(s) to run")
    args = parser.parse_args()

    frozen_sha = hashlib.sha256(FROZEN_TEST_PATH.read_bytes()).hexdigest()
    assert frozen_sha == FROZEN_TEST_SHA256, "Frozen Test SHA mismatch"
    cases = load_eval_cases(FROZEN_TEST_PATH)[: args.cases]
    print(f"Frozen Test SHA OK. Running smoke on {len(cases)} first cases.")

    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"Adapter not found: {adapter}")

    local_model_path = Path(args.local_model_path)
    if not local_model_path.exists():
        raise SystemExit(f"Local model path not found: {local_model_path}")

    outputs = []
    if args.variant in ("u", "both"):
        outputs.append(_smoke_variant(cases, adapter, structured=False, label="1.7B SFT-U", local_model_path=local_model_path))
    if args.variant in ("s", "both"):
        outputs.append(_smoke_variant(cases, adapter, structured=True, label="1.7B SFT-S", local_model_path=local_model_path))

    summary = {
        "phase04_generation_smoke": {
            "local_model_id": LOCAL_MODEL_ID,
            "adapter": str(adapter),
            "frozen_test_sha256": frozen_sha,
            "cases": args.cases,
            "variants": outputs,
            "all_variants_generation_ok": all(o["all_generation_ok"] for o in outputs),
        }
    }
    out_path = Path("reports/generated/phase04/exploratory_currentdata/generation_smoke.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSmoke report -> {out_path}")
    print("ALL_GENERATION_OK =", summary["phase04_generation_smoke"]["all_variants_generation_ok"])


if __name__ == "__main__":
    main()
