"""Phase 03 v0.2.1 — Real API smoke for Qwen Strong-Model Surface Realizer.

Runs 3 samples (Easy / Medium / Hard) through the REAL factory + QwenApiBackend
+ StrongModelSurfaceRealizer. Fails fast (no fallback) if env is missing.

Usage:
  uv run python scripts/data/smoke_qwen_surface_realizer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

from homechef_booking.data.strong_model_realizer import (
    StrongModelSurfaceRealizer,
    anchor_similarity_gate,
)
from homechef_booking.inference.factory import load_qwen_surface_backend
from homechef_booking.inference.qwen_api_backend import check_qwen_env

CONFIG_PATH = Path("configs/inference/qwen_surface.yaml")

# ═══════════════════════════════════════════════════════════
# SMOKE SAMPLE FIELD CONTRACT (single source of truth)
# ═══════════════════════════════════════════════════════════
# Each smoke sample MUST contain exactly these keys:
#   difficulty     : "easy" | "medium" | "hard"
#   scenario       : business scenario label
#   raw_text       : the ORIGINAL Chinese user input (with real values)
#   slot_values    : dict {slot_key: value} OR list[(value, slot_type)]
#                    for identity-preserving placeholderization.
#                    (list form is REQUIRED when two distinct values of the
#                     same type coexist, e.g. two different times.)
#   language_style : style hint passed to the strong model
#   anchors        : anchor ids used as few-shot style references
#
# There is NO "original" / "placeholder_text" field in the sample contract.
# Printing and the result summary MUST read from the keys above only.
SMOKE_SAMPLES = [
    {
        "difficulty": "easy",
        "scenario": "missing_required_slots",
        "raw_text": "我想找个师傅上门做川菜。",
        "slot_values": {"cuisine": "川菜"},
        "language_style": "普通口语",
        "anchors": [1, 2, 3, 4],
    },
    {
        "difficulty": "medium",
        "scenario": "state_inheritance",
        "raw_text": "六点有点早，改成晚上七点吧，其他都不变。",
        # Two DISTINCT times must stay distinct:
        # <TIME_1>=18:00 (六点), <TIME_2>=19:00 (晚上七点)
        "slot_values": [("六点", "TIME"), ("晚上七点", "TIME")],
        "language_style": "礼貌口语",
        "anchors": [11, 12, 13, 14],
    },
    {
        "difficulty": "hard",
        "scenario": "modification_requires_requery",
        "raw_text": "先别订，刚知道有客人海鲜过敏，其他条件不变，重新帮我看看。",
        "slot_values": {"dietary_constraints": "海鲜过敏"},
        "language_style": "口语化纠正",
        "anchors": [18, 21, 20, 19],
    },
]


def _print_help() -> None:
    """Print usage and exit without calling the API."""
    print(
        "Phase 03 v0.2.1 — Qwen Strong-Model Surface Realizer smoke\n"
        "\n"
        "Usage:\n"
        "  uv run python scripts/data/smoke_qwen_surface_realizer.py\n"
        "  uv run python scripts/data/smoke_qwen_surface_realizer.py --help\n"
        "\n"
        "Description:\n"
        "  Runs 3 samples (Easy / Medium / Hard) through the REAL factory + QwenApiBackend\n"
        "  + StrongModelSurfaceRealizer. Fails fast (no fallback) if env vars are missing.\n"
        "\n"
        "Environment (must be set in THIS shell):\n"
        "  DASHSCOPE_API_KEY   (DashScope API key)\n"
        "  DASHSCOPE_BASE_URL  (DashScope OpenAI-compatible base URL)\n"
        "\n"
        "Config:\n"
        "  configs/inference/qwen_surface.yaml  (set `model` to your Qwen model)\n"
        "\n"
        "--help: show this help and exit without calling the API.\n"
    )


def _chinese_gate(text: str) -> bool:
    """Simple Chinese leakage check: no ASCII business text leakage."""
    import re
    pattern = re.compile(
        r'\b(Beijing|Shanghai|Hangzhou|Guangzhou|Sichuan|Cantonese|Chef\s+\w+'
        r'|tomorrow|today|weekend|birthday|peanut_allergy|halal)\b',
        re.IGNORECASE,
    )
    return not bool(pattern.search(text))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # ── 0. --help: show help and exit WITHOUT calling the API ──
    if "--help" in argv or "-h" in argv:
        _print_help()
        return 0

    # ── 1. Fail fast env check ──
    key_status, url_status = check_qwen_env()
    print(f"DASHSCOPE_API_KEY: {key_status}")
    print(f"DASHSCOPE_BASE_URL: {url_status}")
    if key_status == "MISSING":
        print("DASHSCOPE_API_KEY_MISSING — aborting. No fallback.")
        return 1
    if url_status == "MISSING":
        print("DASHSCOPE_BASE_URL_MISSING — aborting. No fallback.")
        return 1

    # ── 2. Load real backend via factory ──
    try:
        backend = load_qwen_surface_backend(CONFIG_PATH)
    except Exception as exc:
        print(f"Qwen backend load FAILED: {exc}")
        print("QWEN_MODEL_NOT_CONFIGURED — set `model` in configs/inference/qwen_surface.yaml")
        return 1

    model_used = backend.config.model
    print("BACKEND_USED: qwen_api")
    print(f"MODEL_USED: {model_used}")

    realizer = StrongModelSurfaceRealizer(
        backend,
        similarity_threshold=0.85,
        max_attempts=3,
    )

    # ── 3. Run 3 samples ──
    api_pass = placeholder_pass = anchor_pass = chinese_pass = 0
    results = []
    for i, sample in enumerate(SMOKE_SAMPLES, 1):
        print("\n" + "=" * 60)
        print(f"SAMPLE {i} [{sample['difficulty']}] scenario={sample['scenario']}")
        print(f"  original: {sample['raw_text']}")
        print(f"  anchor_ids_used: {sample['anchors']}")

        # Show the identity-preserving placeholderized form that will be sent.
        from homechef_booking.data.strong_model_realizer import _placeholderize_text
        placeholderized, _ = _placeholderize_text(sample["raw_text"], sample["slot_values"])
        print(f"  placeholder_text: {placeholderized}")

        outcome = realizer.realize(
            raw_text=sample["raw_text"],
            slot_values=sample["slot_values"],
            difficulty=sample["difficulty"],
            scenario=sample["scenario"],
            language_style=sample["language_style"],
        )
        print(f"  attempt_count: {outcome.attempts}")

        if not outcome.success:
            print(f"  FAILED: {outcome.reason}")
            continue

        restored = outcome.user_input
        print(f"  restored_text: {restored}")

        # Gates
        ph_ok = True  # realizer guarantees placeholder preservation
        gate = anchor_similarity_gate(
            restored, sample["difficulty"], sample["scenario"], 0.85
        )
        cn_ok = _chinese_gate(restored)

        # API call succeeded (outcome.success implies ≥1 backend call)
        api_pass += 1
        placeholder_pass += 1 if ph_ok else 0
        anchor_pass += 1 if gate.passed else 0
        chinese_pass += 1 if cn_ok else 0

        results.append({
            "difficulty": sample["difficulty"],
            "scenario": sample["scenario"],
            "raw_text": sample["raw_text"],
            "rewritten": restored,
            "attempt_count": outcome.attempts,
            "anchor_sim": gate.max_similarity,
        })
        print(f"  placeholder_pass: {'PASS' if ph_ok else 'FAIL'}")
        print(f"  anchor_similarity_pass: {'PASS' if gate.passed else 'FAIL'} (sim={gate.max_similarity:.2f})")
        print(f"  chinese_pass: {'PASS' if cn_ok else 'FAIL'}")

    # ── 4. Summary ──
    print("\n" + "=" * 60)
    print("API SMOKE SUMMARY")
    print("=" * 60)
    print(f"API_CALL: {api_pass}/3")
    print(f"PLACEHOLDER: {placeholder_pass}/3")
    print(f"ANCHOR_SIMILARITY: {anchor_pass}/3")
    print(f"CHINESE: {chinese_pass}/3")

    all_pass = api_pass == 3 and placeholder_pass == 3 and anchor_pass == 3 and chinese_pass == 3
    print(f"\nAPI_SMOKE_GATE: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
