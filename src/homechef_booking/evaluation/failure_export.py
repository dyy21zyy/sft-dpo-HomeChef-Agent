"""Phase 02 failure analysis export for diagnostic dev and error triage."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def export_failure_report(case_results_path: Path, output_path: Path, suite_id: str = "") -> Path:
    case_results = json.loads(case_results_path.read_text(encoding="utf-8"))
    failed = [case for case in case_results if not case.get("effective_pass", False)]
    critical = [case for case in case_results if case.get("critical_error", False)]
    failed_ids = [case["id"] for case in failed]
    critical_ids = [case["id"] for case in critical]
    failure_by_tag: dict[str, list[str]] = {}
    for case in critical:
        for tag in case.get("critical_error_tags", []):
            failure_by_tag.setdefault(tag, []).append(case["id"])
    report = {
        "suite_id": suite_id or case_results_path.stem,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_cases": len(case_results),
            "total_failed": len(failed),
            "total_critical": len(critical),
            "failed_case_ids": failed_ids,
            "critical_case_ids": critical_ids,
        },
        "failure_by_critical_tag": failure_by_tag,
        "failed_cases": [{
            "id": case["id"],
            "protocol_pass": case.get("protocol_pass"),
            "effective_pass": case.get("effective_pass"),
            "task_correctness": case.get("task_correctness"),
            "critical_error": case.get("critical_error"),
            "critical_error_tags": case.get("critical_error_tags", []),
            "structured_score": case.get("structured_score"),
            "reply_score": case.get("reply_score"),
        } for case in failed],
        "critical_error_cases": [{
            "id": case["id"],
            "critical_error_tags": case.get("critical_error_tags", []),
            "task_correctness": case.get("task_correctness"),
        } for case in critical],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
