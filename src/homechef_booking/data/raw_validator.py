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


class RawValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    total: int
    valid: int
    error_count: int
    errors: list[ValidationErrorRecord] = Field(default_factory=list)


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


def validate_raw_jsonl(path: Path) -> RawValidationReport:
    errors: list[ValidationErrorRecord] = []
    total = 0
    valid = 0
    seen_ids: set[str] = set()
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
        except (ValidationError, ValueError) as exc:
            errors.append(ValidationErrorRecord(row=row_idx, sample_id=sample_id, path="schema", message=str(exc)))
            continue
        valid += 1
    return RawValidationReport(path=str(path), total=total, valid=valid - len(errors), error_count=len(errors), errors=errors)


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
