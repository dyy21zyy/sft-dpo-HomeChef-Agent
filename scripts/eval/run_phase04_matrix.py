"""Phase 04 12-cell Frozen Test Evaluation Matrix — build & validate.

Builds the 12-cell exploratory_currentdata matrix configs:

    6 checkpoints x 2 runtime modes (U/S) = 12 cells
    1.7B/4B x {SFT, DPO-β.1, DPO-β.3} x {U, S}

Structured/Unstructured are runtime evaluation modes, NOT separate checkpoints.
Each checkpoint appears exactly twice (one U, one S), sharing base model +
adapter; they differ ONLY in use_structured_output.

Writes eval config YAMLs into configs/evaluation/phase04/exploratory_matrix/.
Does NOT run inference / train / generate reports.

Usage:
    python scripts/eval/run_phase04_matrix.py [--write]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from homechef_booking.training.formal_matrix import (
    build_phase04_exploratory_matrix,
    exploratory_matrix_to_yaml,
    validate_exploratory_matrix,
)

_EVAL_DIR = Path("configs/evaluation/phase04/exploratory_matrix")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase04 12-cell exploratory eval matrix configs")
    parser.add_argument("--write", action="store_true",
                        help="Write the 12 eval config YAMLs to configs/")
    args = parser.parse_args()

    runs = build_phase04_exploratory_matrix()
    errors = validate_exploratory_matrix(runs)
    if errors:
        print("Phase04 exploratory eval matrix validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"Phase04 eval matrix OK: {len(runs)} runs "
          f"(6 checkpoints x 2 runtime modes; experiment_class=exploratory_currentdata)")
    for run in runs:
        print(f"  {run.run_id:<26} model={run.model_id} stage={run.stage} "
              f"beta={run.pref_beta} variant={run.variant} "
              f"structured={run.use_structured_output} adapter={run.adapter_name_or_path}")

    if args.write:
        _EVAL_DIR.mkdir(parents=True, exist_ok=True)
        for run, entry in zip(runs, exploratory_matrix_to_yaml(runs), strict=True):
            out = _EVAL_DIR / f"{run.run_id}.yaml"
            out.write_text(yaml.dump(entry, sort_keys=False, allow_unicode=True), encoding="utf-8")
            print(f"  wrote {out}")
    print("NOTE: no inference / training / reports generated.")


if __name__ == "__main__":
    main()
