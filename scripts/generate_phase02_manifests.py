"""Generate Phase 02 suite manifests after ChatGPT approval."""
import json
from pathlib import Path
from datetime import datetime, timezone
from homechef_booking.evaluation.suite_manifest import compute_sha256

FROZEN_JSONL = Path("data/eval/frozen_test.jsonl")
FROZEN_MANIFEST = Path("data/eval/frozen_test.manifest.json")
DIAG_JSONL = Path("data/dev/diagnostic_dev.jsonl")
DIAG_MANIFEST = Path("data/dev/diagnostic_dev.manifest.json")

now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

frozen_sha = compute_sha256(FROZEN_JSONL)
frozen_manifest = {
    "suite_id": "phase02_frozen_test_v1",
    "contract_id": "homechef-booking-v1",
    "case_file": "data/eval/frozen_test.jsonl",
    "case_count": 120,
    "sha256": frozen_sha,
    "frozen": True,
    "created_at": now,
    "approved_by": "ChatGPT",
    "notes": "Phase 02 Frozen Test suite for formal model selection. 120 HomeChef-contract eval cases. Must not be used for training.",
}
FROZEN_MANIFEST.write_text(json.dumps(frozen_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Frozen manifest: sha256={frozen_sha[:12]}... written to {FROZEN_MANIFEST}")

diag_sha = compute_sha256(DIAG_JSONL)
diag_manifest = {
    "suite_id": "phase02_diagnostic_dev_v1",
    "contract_id": "homechef-booking-v1",
    "case_file": "data/dev/diagnostic_dev.jsonl",
    "case_count": 80,
    "sha256": diag_sha,
    "frozen": False,
    "versioned": True,
    "created_at": now,
    "approved_by": "ChatGPT",
    "notes": "Phase 02 Diagnostic Dev suite for error analysis. 80 HomeChef-contract eval cases. May be versioned for Phase 06 iteration.",
}
DIAG_MANIFEST.write_text(json.dumps(diag_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Diagnostic manifest: sha256={diag_sha[:12]}... written to {DIAG_MANIFEST}")
