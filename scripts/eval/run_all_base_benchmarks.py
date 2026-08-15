"""Run Phase 02 M0 base benchmarks sequentially (llama.cpp CPU path).

Usage: uv run python scripts/eval/run_all_base_benchmarks.py
Outputs results to reports/generated/phase02/

This is the frozen-only matrix: 3 models × Frozen Test = 3 runs.
Diagnostic Dev is preserved but excluded from current formal matrix.
"""

import subprocess
import sys
from pathlib import Path

# Phase 02 Frozen Test only — 3 models
RUNS = [
    "configs/evaluation/phase02_base_0_6b_frozen.yaml",
    "configs/evaluation/phase02_base_1_7b_frozen.yaml",
    "configs/evaluation/phase02_base_4b_frozen.yaml",
]

for i, config_path in enumerate(RUNS, 1):
    name = Path(config_path).stem
    print(f"\n{'='*60}")
    print(f"[{i}/{len(RUNS)}] Running: {name}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        ["uv", "run", "homechef-eval", "--config", config_path],
        capture_output=True, text=True, cwd=Path.cwd(),
    )
    if result.returncode != 0:
        print(f"FAILED with exit code {result.returncode}")
        print(result.stderr)
        sys.exit(1)
    for line in result.stdout.splitlines():
        line = line.strip()
        if line and ":" in line and not line.startswith("C:"):
            print(f"  {line}")
    print("  COMPLETE")

print(f"\n{'='*60}")
print("All 3 benchmark runs complete.")
print("Generating combined summary from saved artifacts...")
print(f"{'='*60}")

summary_result = subprocess.run(
    [
        "uv", "run", "python", "scripts/eval/run_base_benchmark.py",
        "--config", RUNS[0],
        "--config", RUNS[1],
        "--config", RUNS[2],
    ],
    capture_output=True, text=True, cwd=Path.cwd(),
)
print(summary_result.stdout)
if summary_result.returncode != 0:
    print(summary_result.stderr)
    sys.exit(1)

print("\nDONE. Check reports/generated/phase02/ for outputs.")
