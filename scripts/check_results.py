"""Check existing benchmark results."""
import json
from pathlib import Path


def check(path):
    sc = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"  Total cases: {sc['metrics']['total_cases']}")
    print(f"  Protocol pass: {sc['metrics']['protocol_pass_rate']}")
    print(f"  Effective pass: {sc['metrics']['effective_pass_rate']}")
    print(f"  Mean task: {sc['metrics']['mean_task_correctness']}")
    print(f"  Critical error: {sc['metrics']['critical_error_rate']}")

base = Path("reports/generated/phase02")
for run_dir in sorted(base.rglob("phase02_*_scorecard.json")):
    print(f"\n{run_dir.parent.name}/{run_dir.name}:")
    check(run_dir)
