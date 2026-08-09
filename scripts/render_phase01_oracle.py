"""Generate phase01_oracle.json from the frozen fixture spec."""

import json
from pathlib import Path

FROZEN_SPEC_PATH = Path("../.plans/sft-dpo-HomeChef-Agent/phase01_frozen_fixture_spec.json")
ORACLE_OUT = Path("tests/fixtures/evaluation/phase01_oracle.json")


def main() -> None:
    spec = json.loads(FROZEN_SPEC_PATH.read_text(encoding="utf-8"))
    oracle = {}
    for case in spec["cases"]:
        oracle[case["id"]] = {"expected_pass": case["oracle_pass"], "expected_critical_error": case["oracle_critical_error"]}
    ORACLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    ORACLE_OUT.write_text(json.dumps(oracle, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    print(f"Rendered oracle for {len(oracle)} cases to {ORACLE_OUT}")


if __name__ == "__main__":
    main()
