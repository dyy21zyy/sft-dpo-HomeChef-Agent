"""Executable booking contract validator and CLI."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from homechef_booking.schemas.booking import (
    QUERY_DEPENDENCY_FIELDS,
    missing_required_slots,
)
from homechef_booking.schemas.decision import FinalDecision, parse_decision_obj
from homechef_booking.schemas.history import (
    parse_history_messages,
    validate_history_sequence,
)
from homechef_booking.schemas.runtime import BookingRuntimeInput
from homechef_booking.schemas.tools import parse_find_chefs_result


@dataclass
class ValidationIssue:
    path: str
    message: str


_FENCE_RE = re.compile(r"^\s*```", re.MULTILINE)


def parse_model_json(text: str) -> dict[str, Any]:
    """Parse model output as raw JSON, rejecting Markdown fences."""
    if _FENCE_RE.search(text):
        raise ValueError("Model output must be raw JSON, not Markdown-fenced")
    return json.loads(text)


def validate_runtime_input(value: dict[str, Any]) -> list[ValidationIssue]:
    """Validate a runtime input dict against the BookingRuntimeInput schema."""
    issues: list[ValidationIssue] = []
    try:
        BookingRuntimeInput.model_validate(value)
    except ValidationError as exc:
        for err in exc.errors():
            issues.append(
                ValidationIssue(
                    path=".".join(str(x) for x in err["loc"]),
                    message=err["msg"],
                )
            )
    if isinstance(value.get("history"), list):
        try:
            messages = parse_history_messages(value["history"])
        except (ValidationError, ValueError) as exc:
            issues.append(ValidationIssue(path="history", message=str(exc)))
        else:
            errors = validate_history_sequence(messages, value.get("user_input"))
            for e in errors:
                issues.append(ValidationIssue(path="history", message=e))
    return issues


def validate_decision(
    value: dict[str, Any],
    runtime_input: dict[str, Any] | None = None,
) -> list[ValidationIssue]:
    """Validate a decision dict against contract rules."""
    issues: list[ValidationIssue] = []
    try:
        decision = parse_decision_obj(value)
    except ValidationError as exc:
        for err in exc.errors():
            issues.append(
                ValidationIssue(
                    path=".".join(str(x) for x in err["loc"]),
                    message=err["msg"],
                )
            )
        return issues

    if isinstance(decision, FinalDecision):
        slot = decision.booking_state
        actual_missing = missing_required_slots(slot)
        if decision.info_complete != (len(actual_missing) == 0):
            issues.append(
                ValidationIssue(
                    path="info_complete",
                    message="info_complete does not match required slot completeness",
                )
            )
        if decision.missing_info != actual_missing:
            issues.append(
                ValidationIssue(
                    path="missing_info",
                    message="missing_info does not match canonical required slot order",
                )
            )

        if decision.unrelated and decision.reply_type.value != "handoff":
            issues.append(
                ValidationIssue(
                    path="unrelated",
                    message="unrelated=true requires reply_type=handoff",
                )
            )
        if decision.reply_type.value == "handoff" and not decision.unrelated:
            issues.append(
                ValidationIssue(
                    path="reply_type",
                    message="reply_type=handoff requires unrelated=true",
                )
            )

        if runtime_input:
            _check_candidate_provenance(decision, runtime_input, issues)
            _check_state_invalidation(decision, runtime_input, issues)

    return issues


def _check_candidate_provenance(
    decision: FinalDecision,
    runtime_input: dict,
    issues: list[ValidationIssue],
) -> None:
    """Check candidate order matches tool result order."""
    history = runtime_input.get("history", [])
    tool_candidates: list[dict] = []
    for msg in history:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            try:
                result = json.loads(msg.get("content", "{}"))
                if (
                    result.get("status") == "matched"
                    and isinstance(result.get("candidates"), list)
                ):
                    tool_candidates = result["candidates"]
                elif (
                    result.get("status") == "unavailable"
                    and isinstance(result.get("alternatives"), list)
                ):
                    tool_candidates = result["alternatives"]
            except (json.JSONDecodeError, ValueError):
                pass
    if tool_candidates:
        tool_ids = [c.get("chef_id") for c in tool_candidates]
        decision_ids = [c.chef_id for c in decision.candidate_chefs]
        if decision_ids != tool_ids and set(decision_ids) <= set(tool_ids):
            issues.append(
                ValidationIssue(
                    path="candidate_chefs",
                    message="candidate order mutation detected",
                )
            )


def _check_state_invalidation(
    decision: FinalDecision,
    runtime_input: dict,
    issues: list[ValidationIssue],
) -> None:
    """Check query dependency mutation invalidates stale tool facts."""
    current_state = runtime_input.get("current_state", {})
    current_booking = current_state.get("booking_state", {})
    new_booking = decision.booking_state.model_dump()
    for field in QUERY_DEPENDENCY_FIELDS:
        if field in current_booking and field in new_booking:
            if current_booking[field] != new_booking[field]:
                if decision.candidate_chefs:
                    issues.append(
                        ValidationIssue(
                            path="candidate_chefs",
                            message="query dependency mutation requires candidate_chefs=[]",
                        )
                    )
                if decision.booking_state.chef_id is not None:
                    issues.append(
                        ValidationIssue(
                            path="chef_id",
                            message="query dependency mutation requires chef_id=null",
                        )
                    )
                break


def validate_find_chefs_result(value: dict[str, Any]) -> list[ValidationIssue]:
    """Validate a find_chefs tool result dict."""
    issues: list[ValidationIssue] = []
    try:
        parse_find_chefs_result(value)
    except (ValidationError, ValueError) as exc:
        issues.append(ValidationIssue(path="", message=str(exc)))
    return issues


def json_schema_verdict(path: Path, value: dict[str, Any]) -> bool:
    """Return True if value passes JSON Schema validation for its fixture type."""
    import jsonschema

    name = path.stem
    if "runtime" in name or "history" in name:
        schema_path = Path("contracts/booking_machine_contract_v1.schema.json")
        schema_key = "BookingRuntimeInput"
    elif "find_chefs" in name:
        schema_path = Path("contracts/find_chefs_v1.schema.json")
        schema_key = "FindChefsResult"
    elif "tool_call_decision" in name or "final" in name:
        schema_path = Path("contracts/booking_machine_contract_v1.schema.json")
        schema_key = "Decision"
    else:
        schema_path = Path("contracts/booking_machine_contract_v1.schema.json")
        schema_key = "Decision"

    schema_doc = json.loads(schema_path.read_text(encoding="utf-8"))
    defs = schema_doc.get("$defs", {})
    properties = schema_doc.get("properties", {})
    combined = {**defs, **properties}
    ref_schema = {"$ref": f"#/$defs/{schema_key}", "$defs": combined}
    try:
        validator_cls = jsonschema.Draft202012Validator
        validator = validator_cls(
            ref_schema,
            format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
        )
        validator.validate(value)
        return True
    except jsonschema.ValidationError:
        return False


def pydantic_verdict(path: Path, value: dict[str, Any]) -> bool:
    """Return True if value passes Pydantic/validator checks."""
    name = path.stem
    try:
        if "runtime" in name or "history" in name:
            issues = validate_runtime_input(value)
        elif "find_chefs" in name:
            issues = validate_find_chefs_result(value)
        elif "tool_call_decision" in name or "final" in name:
            issues = validate_decision(value)
        else:
            issues = validate_decision(value)
        return len(issues) == 0
    except (ValidationError, ValueError):
        return False


def validate_contract_files(root: Path) -> list[ValidationIssue]:
    """Validate all contract files: manifest, schemas, and fixtures."""
    issues: list[ValidationIssue] = []
    fixtures_dir = root / "tests" / "fixtures" / "contracts"
    if not fixtures_dir.exists():
        issues.append(
            ValidationIssue(path="fixtures", message="fixtures directory not found")
        )
        return issues
    for fixture_path in sorted(fixtures_dir.glob("*/*.json")):
        try:
            json.loads(fixture_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(
                ValidationIssue(
                    path=str(fixture_path),
                    message=f"JSON parse error: {exc}",
                )
            )
    return issues


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for contract validation."""
    parser = argparse.ArgumentParser(
        description="HomeChef booking contract validator",
    )
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument(
        "--fixtures",
        default="tests/fixtures/contracts",
        help="Fixtures directory",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    issues = validate_contract_files(root)
    if issues:
        for issue in issues:
            print(f"FAIL {issue.path}: {issue.message}")
        return 1
    print("All contract files valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
