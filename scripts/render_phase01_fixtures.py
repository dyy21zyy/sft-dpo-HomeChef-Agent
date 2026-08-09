"""Mechanically render phase01 fixtures from the frozen spec.
Usage: uv run python scripts/render_phase01_fixtures.py
"""

import json
from pathlib import Path

FROZEN_SPEC_PATH = Path("../.plans/sft-dpo-HomeChef-Agent/phase01_frozen_fixture_spec.json")
CASES_OUT = Path("tests/fixtures/evaluation/phase01_mock_cases.jsonl")
PREDICTIONS_OUT = Path("tests/fixtures/evaluation/phase01_mock_predictions.json")


def main() -> None:
    spec = json.loads(FROZEN_SPEC_PATH.read_text(encoding="utf-8"))

    # Render cases.jsonl
    lines = []
    for case in spec["cases"]:
        line = {
            "id": case["id"],
            "output_kind": case["output_kind"],
            "conversation_kind": case["conversation_kind"],
            "input": case["input"],
            "expected": case["expected"],
            "assertions": case["assertions"],
            "tags": case["tags"],
            "reply_expectations": case["reply_expectations"],
        }
        if case.get("chain_id"):
            line["chain_id"] = case["chain_id"]
        if case.get("step") is not None:
            line["step"] = case["step"]
        lines.append(json.dumps(line, ensure_ascii=False, sort_keys=True))

    CASES_OUT.parent.mkdir(parents=True, exist_ok=True)
    CASES_OUT.write_text("\n".join(lines), encoding="utf-8")

    # Render predictions.json
    predictions = {}
    for case in spec["cases"]:
        if "mock_prediction_raw" in case:
            predictions[case["id"]] = {"raw_text": case["mock_prediction_raw"]}
        else:
            predictions[case["id"]] = {"raw_text": json.dumps(case["mock_prediction"], ensure_ascii=False, sort_keys=True)}

    PREDICTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_OUT.write_text(json.dumps(predictions, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

    print(f"Rendered {len(lines)} cases to {CASES_OUT}")
    print(f"Rendered {len(predictions)} predictions to {PREDICTIONS_OUT}")


if __name__ == "__main__":
    main()
