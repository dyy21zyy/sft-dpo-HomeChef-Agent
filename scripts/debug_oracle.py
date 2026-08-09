"""Debug oracle mismatches."""
import json
from pathlib import Path

PREDS = Path("tests/fixtures/evaluation/phase01_mock_predictions.json")
CASES = Path("tests/fixtures/evaluation/phase01_mock_cases.jsonl")

predictions = json.loads(PREDS.read_text(encoding="utf-8"))

problem_ids = ["case_specific_unavailable", "case_candidate_selection", "case_fabricated_chef", "case_unavailable_reversal", "case_unauthorized_booking"]

for pid in problem_ids:
    raw = predictions[pid]["raw_text"]
    pred = json.loads(raw)
    print(f"=== {pid} ===")
    print(f"  action: {pred.get('action')}")
    if pred.get("action") == "final":
        bs = pred.get("booking_state", {})
        print(f"  chef_id: {bs.get('chef_id')}")
        print(f"  chef_name: {bs.get('chef_name')}")
        print(f"  reply: {pred.get('reply', '')[:100]}")
        print(f"  reply_type: {pred.get('reply_type')}")
        print(f"  confirmation: {bs.get('confirmation')}")
    print()
