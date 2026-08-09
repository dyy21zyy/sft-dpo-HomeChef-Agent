"""Debug oracle mismatches - file output."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PREDS = Path("tests/fixtures/evaluation/phase01_mock_predictions.json")
predictions = json.loads(PREDS.read_text(encoding="utf-8"))

problem_ids = ["case_specific_unavailable", "case_candidate_selection", "case_fabricated_chef", "case_unavailable_reversal", "case_unauthorized_booking"]

for pid in problem_ids:
    raw = predictions[pid]["raw_text"]
    pred = json.loads(raw)
    bs = pred.get("booking_state", {})
    print(f"{pid}: action={pred.get('action')} chef_id={bs.get('chef_id')} reply_type={pred.get('reply_type')}")
    reply = pred.get('reply', '') or ''
    print(f"  reply={reply[:100]}")
    print(f"  confirmation={bs.get('confirmation')}")
    print(f"  chef_name={bs.get('chef_name')}")
