"""Self-check: validate all 19 cases, predictions, and oracle from rendered fixtures."""

import json
from pathlib import Path

from homechef_booking.evaluation.sample import EvalCase

CASES_PATH = Path("tests/fixtures/evaluation/phase01_mock_cases.jsonl")
PREDICTIONS_PATH = Path("tests/fixtures/evaluation/phase01_mock_predictions.json")
ORACLE_PATH = Path("tests/fixtures/evaluation/phase01_oracle.json")

# 1. Validate all cases load
print("=== CASE VALIDATION ===")
cases = []
with open(CASES_PATH, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        case = EvalCase.model_validate(json.loads(line))
        cases.append(case)
        print(f"  OK line {i}: {case.id} (output_kind={case.output_kind}, action={case.expected.action})")

print(f"\nTotal cases: {len(cases)}")

# 2. Check no duplicate IDs
ids = [c.id for c in cases]
assert len(ids) == len(set(ids)), "DUPLICATE IDs!"
print("No duplicate IDs: PASS")

# 3. Check output_kind matches expected.action
for c in cases:
    assert c.output_kind == c.expected.action, f"output_kind mismatch for {c.id}: {c.output_kind} vs {c.expected.action}"
print("output_kind matches expected.action: PASS")

# 4. Validate predictions
print("\n=== PREDICTIONS VALIDATION ===")
predictions = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
case_ids = {c.id for c in cases}
pred_ids = set(predictions.keys())
print(f"Prediction IDs: {len(pred_ids)}")
assert case_ids == pred_ids, f"Mismatch: cases without predictions: {case_ids - pred_ids}, predictions without cases: {pred_ids - case_ids}"
print("Case-Prediction ID match: PASS")

# 5. Validate oracle
print("\n=== ORACLE VALIDATION ===")
oracle = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
oracle_ids = {c["id"] for c in oracle["cases"]}
assert case_ids == oracle_ids, f"Mismatch: cases without oracle: {case_ids - oracle_ids}, oracle without cases: {oracle_ids - case_ids}"
print("Case-Oracle ID match: PASS")

# 6. Tool spec conformance: every case with available_tools must pass validation
print("\n=== TOOL SPEC CONFORMANCE ===")
for c in cases:
    for tool in c.input.available_tools:
        props = tool.function.parameters.properties
        for key in ["chef_name", "service_date", "start_time", "people", "address", "cuisine", "budget_min", "budget_max", "menu", "ingredient_purchase", "dietary_constraints", "occasion"]:
            assert key in props, f"{c.id}: missing property {key}"
    if c.input.available_tools:
        print(f"  {c.id}: tool spec valid (12 keys present)")
print("Tool spec conformance: PASS")

# 7. Count: must be exactly 19
assert len(cases) == 19, f"Expected 19 cases, got {len(cases)}"
print(f"\nCase count: {len(cases)} = 19: PASS")

print("\n=== ALL CHECKS PASSED ===")
