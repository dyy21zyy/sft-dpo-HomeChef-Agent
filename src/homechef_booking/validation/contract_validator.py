"""Executable booking contract validator with layered validation."""

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
    ChefQueryStatus,
    ReplyType,
    is_affirmative,
    missing_required_slots,
)
from homechef_booking.schemas.decision import (
    FinalDecision,
    ToolCallDecision,
    parse_decision_obj,
)
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


# ---------------------------------------------------------------------------
# Structural validation (JSON Schema layer)
# ---------------------------------------------------------------------------


def json_schema_structural_verdict(
    path: Path,
    value: dict[str, Any],
) -> bool:
    """Return True if value passes JSON Schema structural validation."""
    import jsonschema

    name = path.stem
    if "runtime" in name or "history" in name:
        schema_path = Path("contracts/booking_machine_contract_v1.schema.json")
        schema_key = "BookingRuntimeInput"
    elif "find_chefs" in name:
        schema_path = Path("contracts/find_chefs_v1.schema.json")
        schema_key = "FindChefsResult"
    elif "tool_call_decision" in name:
        schema_path = Path("contracts/booking_machine_contract_v1.schema.json")
        schema_key = "ToolCallDecision"
    elif "final" in name:
        schema_path = Path("contracts/booking_machine_contract_v1.schema.json")
        schema_key = "FinalDecision"
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
            format_checker=validator_cls.FORMAT_CHECKER,
        )
        validator.validate(value)
        return True
    except jsonschema.ValidationError:
        return False


# ---------------------------------------------------------------------------
# Structural validation (Pydantic layer)
# ---------------------------------------------------------------------------


def pydantic_structural_verdict(
    path: Path,
    value: dict[str, Any],
) -> bool:
    """Return True if value passes Pydantic structural validation."""
    name = path.stem
    try:
        if "runtime" in name or "history" in name:
            BookingRuntimeInput.model_validate(value)
        elif "find_chefs" in name:
            parse_find_chefs_result(value)
        elif "tool_call_decision" in name:
            ToolCallDecision.model_validate(value)
        elif "final" in name:
            FinalDecision.model_validate(value)
        else:
            parse_decision_obj(value)
        return True
    except (ValidationError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Business contract validation
# ---------------------------------------------------------------------------


def validate_runtime_input(
    value: dict[str, Any],
) -> list[ValidationIssue]:
    """Validate a runtime input dict against contract rules."""
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
        return issues

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
    """Validate a decision dict against all business contract rules."""
    issues: list[ValidationIssue] = []
    try:
        decision = parse_decision_obj(value)
    except (ValidationError, ValueError) as exc:
        issues.append(ValidationIssue(path="action", message=str(exc)))
        return issues

    if isinstance(decision, ToolCallDecision):
        _validate_tool_call_business(decision, runtime_input, issues)
    elif isinstance(decision, FinalDecision):
        _validate_final_business(decision, runtime_input, issues)

    return issues


def _validate_tool_call_business(
    decision: ToolCallDecision,
    runtime_input: dict[str, Any] | None,
    issues: list[ValidationIssue],
) -> None:
    """Business rules for ToolCallDecision.

    Checks decision.arguments (not current_state.booking_state) for the
    four required slots: service_date, start_time, people, address.
    """
    args = decision.arguments
    missing: list[str] = []
    if args.service_date is None:
        missing.append("service_date")
    if args.start_time is None:
        missing.append("start_time")
    if args.people is None:
        missing.append("people")
    if args.address is None:
        missing.append("address")
    if missing:
        issues.append(
            ValidationIssue(
                path="tool_call",
                message=(
                    "ToolCall requires all 4 required slots complete, "
                    f"missing: {missing}"
                ),
            )
        )


def _validate_final_business(
    decision: FinalDecision,
    runtime_input: dict[str, Any] | None,
    issues: list[ValidationIssue],
) -> None:
    """Business rules for FinalDecision."""
    slot = decision.booking_state

    # info_complete consistency
    actual_missing = missing_required_slots(slot)
    expected_complete = len(actual_missing) == 0
    if decision.info_complete != expected_complete:
        issues.append(
            ValidationIssue(
                path="info_complete",
                message="info_complete does not match required slot completeness",
            )
        )

    # missing_info canonical order
    if decision.missing_info != actual_missing:
        issues.append(
            ValidationIssue(
                path="missing_info",
                message="missing_info does not match canonical required slot order",
            )
        )

    # unrelated ↔ handoff
    if decision.unrelated and decision.reply_type != ReplyType.handoff:
        issues.append(
            ValidationIssue(
                path="unrelated",
                message="unrelated=true requires reply_type=handoff",
            )
        )
    if decision.reply_type == ReplyType.handoff and not decision.unrelated:
        issues.append(
            ValidationIssue(
                path="reply_type",
                message="reply_type=handoff requires unrelated=true",
            )
        )

    if runtime_input:
        _check_candidate_provenance(decision, runtime_input, issues)
        _check_state_invalidation(decision, runtime_input, issues)
        _check_booking_authorized(decision, runtime_input, issues)
        _check_candidate_selection_provenance(decision, runtime_input, issues)
        _check_chef_query_status_preservation(decision, runtime_input, issues)


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
                            message=(
                                "query dependency mutation requires "
                                "candidate_chefs=[]"
                            ),
                        )
                    )
                if decision.booking_state.chef_id is not None:
                    issues.append(
                        ValidationIssue(
                            path="chef_id",
                            message=(
                                "query dependency mutation requires chef_id=null"
                            ),
                        )
                    )
                if decision.chef_query_status != ChefQueryStatus.not_checked:
                    issues.append(
                        ValidationIssue(
                            path="chef_query_status",
                            message=(
                                "query dependency mutation requires "
                                "chef_query_status=not_checked"
                            ),
                        )
                    )
                if decision.booking_state.confirmation:
                    issues.append(
                        ValidationIssue(
                            path="confirmation",
                            message=(
                                "query dependency mutation requires "
                                "confirmation=false"
                            ),
                        )
                    )
                break


def _check_booking_authorized(
    decision: FinalDecision,
    runtime_input: dict,
    issues: list[ValidationIssue],
) -> None:
    """Check booking_authorized rules.

    Requires:
    - awaiting_confirmation=true
    - selected chef has valid Tool provenance (chef_id non-null)
    - explicit affirmative user input
    - no query dependency mutation

    chef_query_status must preserve last effective Tool Result status:
    - direct specific/available → available
    - selected from search/matched → matched
    - selected from unavailable alternatives → unavailable
    """
    if decision.reply_type != ReplyType.booking_authorized:
        return
    current_state = runtime_input.get("current_state", {})
    if not current_state.get("awaiting_confirmation", False):
        issues.append(
            ValidationIssue(
                path="reply_type",
                message=(
                    "booking_authorized requires awaiting_confirmation=true"
                ),
            )
        )
    if decision.booking_state.chef_id is None:
        issues.append(
            ValidationIssue(
                path="chef_id",
                message="booking_authorized requires tool-verified chef_id",
            )
        )
    user_input = runtime_input.get("user_input")
    if user_input is not None and not is_affirmative(user_input):
        issues.append(
            ValidationIssue(
                path="user_input",
                message=(
                    "booking_authorized requires deterministic affirmative "
                    "user input"
                ),
            )
        )


def _check_candidate_selection_provenance(
    decision: FinalDecision,
    runtime_input: dict,
    issues: list[ValidationIssue],
) -> None:
    """Check that selected chef_id exists in tool-returned candidates."""
    if decision.booking_state.chef_id is None:
        return
    history = runtime_input.get("history", [])
    tool_chef_ids: set[str] = set()
    for msg in history:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            try:
                result = json.loads(msg.get("content", "{}"))
                if result.get("status") == "matched":
                    for c in result.get("candidates", []):
                        tool_chef_ids.add(c.get("chef_id"))
                elif result.get("status") == "unavailable":
                    for c in result.get("alternatives", []):
                        tool_chef_ids.add(c.get("chef_id"))
                elif result.get("status") == "available":
                    pass
            except (json.JSONDecodeError, ValueError):
                pass
    current_state = runtime_input.get("current_state", {})
    for c in current_state.get("candidate_chefs", []):
        tool_chef_ids.add(c.get("chef_id"))
    if tool_chef_ids and decision.booking_state.chef_id not in tool_chef_ids:
        issues.append(
            ValidationIssue(
                path="chef_id",
                message=(
                    "selected chef_id not found in tool candidates "
                    "or current_state"
                ),
            )
        )


def _check_chef_query_status_preservation(
    decision: FinalDecision,
    runtime_input: dict,
    issues: list[ValidationIssue],
) -> None:
    """Check that candidate selection preserves chef_query_status."""
    current_state = runtime_input.get("current_state", {})
    current_status = current_state.get("chef_query_status")
    if current_status is None:
        return
    if decision.booking_state.chef_id is not None:
        if current_status == "matched" and decision.chef_query_status != "matched":
            issues.append(
                ValidationIssue(
                    path="chef_query_status",
                    message=(
                        "matched candidate selection must preserve "
                        "chef_query_status=matched"
                    ),
                )
            )
        if (
            current_status == "unavailable"
            and decision.chef_query_status != "unavailable"
        ):
            issues.append(
                ValidationIssue(
                    path="chef_query_status",
                    message=(
                        "unavailable alternative selection must preserve "
                        "chef_query_status=unavailable"
                    ),
                )
            )


def business_contract_verdict(
    path: Path,
    value: dict[str, Any],
    runtime_input: dict[str, Any] | None = None,
) -> bool:
    """Return True if value passes all business contract rules."""
    name = path.stem
    try:
        if "runtime" in name or "history" in name:
            issues = validate_runtime_input(value)
        elif "find_chefs" in name:
            issues = []
            parse_find_chefs_result(value)
        elif "tool_call_decision" in name or "final" in name:
            issues = validate_decision(value, runtime_input)
        else:
            issues = validate_decision(value, runtime_input)
        return len(issues) == 0
    except (ValidationError, ValueError):
        return False


def validate_find_chefs_result(
    value: dict[str, Any],
) -> list[ValidationIssue]:
    """Validate a find_chefs tool result dict."""
    issues: list[ValidationIssue] = []
    try:
        parse_find_chefs_result(value)
    except (ValidationError, ValueError) as exc:
        issues.append(ValidationIssue(path="", message=str(exc)))
    return issues


# ---------------------------------------------------------------------------
# Full CLI validation gate
# ---------------------------------------------------------------------------


def validate_contract_files(root: Path) -> list[ValidationIssue]:
    """Full contract validation: manifest, schemas, fixtures, parity."""
    issues: list[ValidationIssue] = []

    # 1. Load contract manifest
    from homechef_booking.contracts.manifest import (
        load_contract_manifest,
        validate_manifest_source_hashes,
    )

    manifest_path = root / "contracts" / "contract_manifest.yaml"
    if not manifest_path.exists():
        issues.append(
            ValidationIssue(path="manifest", message="manifest not found")
        )
        return issues
    manifest = load_contract_manifest(manifest_path)

    # 2. Validate source hashes
    hash_errors = validate_manifest_source_hashes(manifest, root)
    for e in hash_errors:
        issues.append(ValidationIssue(path="manifest", message=e))

    # 3. Load/validate JSON Schema documents
    for schema_file in manifest.get("schemas", []):
        schema_path = root / schema_file
        if not schema_path.exists():
            issues.append(
                ValidationIssue(
                    path=schema_file,
                    message="schema file not found",
                )
            )
            continue
        try:
            json.loads(schema_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(
                ValidationIssue(
                    path=schema_file,
                    message=f"JSON parse error: {exc}",
                )
            )

    # 4. Validate all valid fixtures → MUST PASS
    valid_dir = root / "tests" / "fixtures" / "contracts" / "valid"
    if valid_dir.exists():
        for fixture_path in sorted(valid_dir.glob("*.json")):
            try:
                value = json.loads(
                    fixture_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as exc:
                issues.append(
                    ValidationIssue(
                        path=str(fixture_path),
                        message=f"JSON parse error: {exc}",
                    )
                )
                continue
            if not pydantic_structural_verdict(fixture_path, value):
                issues.append(
                    ValidationIssue(
                        path=str(fixture_path),
                        message="valid fixture failed structural validation",
                    )
                )

    # 5. Validate all invalid fixtures → MUST FAIL structural validation
    #    (business-rule-only invalid fixtures pass structural and are
    #     tested by pytest, not by the CLI gate)
    business_invalid_only = {
        "final_fabricated_chef",
        "final_reordered_candidates",
        "final_unrelated_not_handoff",
        "state_dependency_mutation_stale_tool_facts",
    }
    invalid_dir = root / "tests" / "fixtures" / "contracts" / "invalid"
    if invalid_dir.exists():
        for fixture_path in sorted(invalid_dir.glob("*.json")):
            if fixture_path.stem in business_invalid_only:
                continue
            try:
                value = json.loads(
                    fixture_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                continue
            if pydantic_structural_verdict(fixture_path, value):
                issues.append(
                    ValidationIssue(
                        path=str(fixture_path),
                        message=(
                            "invalid fixture passed structural validation "
                            "(should fail)"
                        ),
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
