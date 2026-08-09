"""Run all four Phase 02 M0 base benchmarks sequentially.

Usage: uv run python scripts/eval/run_all_base_benchmarks.py
Outputs results to reports/generated/phase02/
"""

import subprocess
import sys
from pathlib import Path

RUNS = [
    "configs/evaluation/phase02_base_0_6b_frozen.yaml",
    "configs/evaluation/phase02_base_0_6b_diagnostic.yaml",
    "configs/evaluation/phase02_base_1_7b_frozen.yaml",
    "configs/evaluation/phase02_base_1_7b_diagnostic.yaml",
]

env = {"PYTHONIOENCODING": "utf-8", **dict(sys.executable and {} or {})}

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
    # Parse key metrics from stdout
    for line in result.stdout.splitlines():
        line = line.strip()
        if line and ":" in line and not line.startswith("C:"):
            print(f"  {line}")
    print("  COMPLETE")

print(f"\n{'='*60}")
print("All 4 benchmark runs complete.")
print("Now running combined summary...")
print(f"{'='*60}")

summary_result = subprocess.run(
    [
        "uv", "run", "python", "scripts/eval/run_base_benchmark.py",
        "--config", RUNS[0], "--config", RUNS[1],
        "--config", RUNS[2], "--config", RUNS[3],
    ],
    capture_output=True, text=True, cwd=Path.cwd(),
)
print(summary_result.stdout)
if summary_result.returncode != 0:
    print(summary_result.stderr)
    sys.exit(1)

print("\nDONE. Check reports/generated/phase02/ for outputs.")
