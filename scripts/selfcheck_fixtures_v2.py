"""Self-check v2: validate all 19 cases from rendered fixtures."""

import json
from pathlib import Path

from homechef_booking.evaluation.sample import EvalCase

CASES_PATH = Path("tests/fixtures/evaluation/phase01_mock_cases.jsonl")
PREDICTIONS_PATH = Path("tests/fixtures/evaluation/phase01_mock_predictions.json")
ORACLE_PATH = Path("tests/fixtures/evaluation/phase01_oracle.json")

cases = []
with open(CASES_PATH, encoding="utf-8") as f:
    for line in f:
        cases.append(EvalCase.model_validate(json.loads(line)))

print(f"Cases validated: {len(cases)}")

ids = [c.id for c in cases]
assert len(ids) == len(set(ids)), "DUPLICATE"
assert len(cases) == 19, f"Expected 19, got {len(cases)}"

predictions = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
oracle = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
assert set(ids) == set(predictions.keys())
assert set(ids) == {c["id"] for c in oracle["cases"]}

for c in cases:
    assert c.output_kind == c.expected.action, f"{c.id}: output_kind mismatch"

# Tool spec: verify cases with tools have valid specs
tool_cases = [c for c in cases if c.input.available_tools]
for c in tool_cases:
    for tool in c.input.available_tools:
        props = tool.model_dump()["function"]["parameters"]["properties"]
        assert "service_date" in props
        assert "start_time" in props
        assert "people" in props
        assert "address" in props
    print(f"  {c.id}: tool spec OK")

print("ALL CHECKS PASSED")
