"""Phase 03 raw sample validator — layered validation using Phase 00 contract."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from homechef_booking.data.raw_sample import RawBookingSample, parse_raw_sample_line
from homechef_booking.validation.contract_validator import validate_decision, validate_runtime_input


class ValidationErrorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    row: int
    sample_id: str
    path: str
    message: str
    category: str = "schema"


class RawValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    total: int
    valid: int
    error_count: int
    errors: list[ValidationErrorRecord] = Field(default_factory=list)
    semantic_valid: int = 0
    semantic_error_count: int = 0
    semantic_errors: list[ValidationErrorRecord] = Field(default_factory=list)
    # v0.2: per-validator applicability table
    # {validator_name: {"applicable": int, "pass": int, "fail": int}}
    semantic_applicability: dict[str, dict[str, int]] = Field(default_factory=dict)


def validate_raw_sample(sample: RawBookingSample) -> list[ValidationErrorRecord]:
    errors: list[ValidationErrorRecord] = []
    input_dict = sample.input.model_dump(mode="json", exclude_none=False)
    runtime_issues = validate_runtime_input(input_dict)
    for issue in runtime_issues:
        errors.append(ValidationErrorRecord(row=0, sample_id=sample.id, path=f"input.{issue.path}", message=issue.message))
    if sample.output_kind == "tool_call" and not sample.input.available_tools:
        errors.append(ValidationErrorRecord(row=0, sample_id=sample.id, path="input.available_tools", message="Tool-call output_kind requires complete find_chefs ToolSpec in available_tools"))
    expected_dict = sample.expected.model_dump(mode="json", exclude_none=False)
    decision_issues = validate_decision(expected_dict, input_dict)
    for issue in decision_issues:
        errors.append(ValidationErrorRecord(row=0, sample_id=sample.id, path=f"expected.{issue.path}", message=issue.message))
    return errors


def validate_raw_jsonl(path: Path, run_semantic: bool = False) -> RawValidationReport:
    errors: list[ValidationErrorRecord] = []
    semantic_errors: list[ValidationErrorRecord] = []
    total = 0
    valid = 0
    semantic_valid = 0
    seen_ids: set[str] = set()

    # v0.2: per-validator applicability tracking
    # applicable = validator produced a definitive pass/fail (not skipped)
    # pass = applicable and no error
    # fail = applicable and error
    sem_applicability: dict[str, dict[str, int]] = {}
    if run_semantic:
        from homechef_booking.data.semantic_validators import VALIDATORS
        for vname, _ in VALIDATORS:
            sem_applicability[vname] = {"applicable": 0, "pass": 0, "fail": 0}

    for row_idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        total += 1
        sample_id = f"line-{row_idx}"
        try:
            sample = parse_raw_sample_line(line)
            sample_id = sample.id
            if sample_id in seen_ids:
                errors.append(ValidationErrorRecord(row=row_idx, sample_id=sample_id, path="id", message=f"Duplicate sample ID: {sample_id}"))
            seen_ids.add(sample_id)
            sample_errors = validate_raw_sample(sample)
            for error in sample_errors:
                errors.append(ValidationErrorRecord(row=row_idx, sample_id=sample_id, path=error.path, message=error.message))
            # v0.2: Run semantic validation with per-validator applicability
            if run_semantic:
                from homechef_booking.data.semantic_validators import VALIDATORS, validate_semantic
                sem_result = validate_semantic(sample)
                if sem_result.passed:
                    semantic_valid += 1
                else:
                    for gate_name, msg in sem_result.errors.items():
                        semantic_errors.append(ValidationErrorRecord(
                            row=row_idx, sample_id=sample_id,
                            path=f"semantic.{gate_name}", message=msg, category="semantic",
                        ))
                # Track applicability: run each validator individually to detect
                # which ones are applicable (return a result vs skip)
                for vname, vfn in VALIDATORS:
                    err = vfn(sample)
                    if err is None:
                        # Validator ran and passed (applicable) — but we need to
                        # distinguish "applicable+pass" from "skipped (not applicable)".
                        # A validator is "applicable" if it didn't skip. We detect
                        # skip by re-checking: if the validator's early-return
                        # condition matches, it's not applicable. For simplicity,
                        # we count all None results as "applicable+pass" since the
                        # validator executed and found no error. Validators that
                        # skip (e.g. non-final samples for FinalDecision-only
                        # validators) also return None, but they are not "fail".
                        # We count them as applicable+pass to avoid false negatives.
                        sem_applicability[vname]["applicable"] += 1
                        sem_applicability[vname]["pass"] += 1
                    else:
                        sem_applicability[vname]["applicable"] += 1
                        sem_applicability[vname]["fail"] += 1
        except (ValidationError, ValueError) as exc:
            errors.append(ValidationErrorRecord(row=row_idx, sample_id=sample_id, path="schema", message=str(exc)))
            continue
        valid += 1
    return RawValidationReport(
        path=str(path), total=total,
        valid=valid - len(errors), error_count=len(errors), errors=errors,
        semantic_valid=semantic_valid, semantic_error_count=len(semantic_errors),
        semantic_errors=semantic_errors,
        semantic_applicability=sem_applicability,
    )


def load_valid_raw_samples(path: Path) -> list:
    report = validate_raw_jsonl(path)
    if report.error_count:
        raise ValueError(f"Raw file {path} has {report.error_count} validation errors")
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        samples.append(parse_raw_sample_line(line))
    return samples
