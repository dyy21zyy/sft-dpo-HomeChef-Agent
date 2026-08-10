"""Phase 04 Task 2 — Dataset adapter and training preflight validators.

Validates Phase 03 SFT and DPO datasets are trainable inputs and rejects
eval suites (Frozen Test, Diagnostic Dev) and Phase 02 benchmark outputs.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from homechef_booking.data.dataset_schema import DpoPair, SftSample

# Row IDs containing any of these substrings are forbidden as training data
_FORBIDDEN_ID_SUBSTRINGS = [
    "frozen",
    "diagnostic",
    "missing_mock_prediction",
    "phase02",
    "base_benchmark",
]


class DatasetPreflightReport(BaseModel):
    """Result of a training dataset preflight validation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_type: str = Field(description="'sft' or 'dpo'")
    total_rows: int = Field(ge=0)
    expected_count: int = Field(ge=0)
    passed: bool
    errors: list[str] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Convenience: number of validation errors."""
        return len(self.errors)


def _is_forbidden_id(row_id: str) -> bool:
    """Check if a row ID contains any forbidden substring."""
    rid_lower = row_id.lower()
    for substr in _FORBIDDEN_ID_SUBSTRINGS:
        if substr in rid_lower:
            return True
    return False


def validate_sft_for_training(path: Path, expected_count: int) -> DatasetPreflightReport:
    """Validate an SFT JSONL file for training readiness.

    Checks:
    - Every row is a valid SftSample.
    - No row ID contains forbidden substrings (frozen, diagnostic, etc.).
    - Total row count matches expected_count.
    """
    errors: list[str] = []
    total = 0

    if not path.exists():
        return DatasetPreflightReport(
            dataset_type="sft",
            total_rows=0,
            expected_count=expected_count,
            passed=False,
            errors=[f"File not found: {path}"],
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Row {total}: invalid JSON — {e}")
            continue
        try:
            sample = SftSample(**row)
        except ValidationError as e:
            errors.append(f"Row {total}: SftSample validation error — {e}")
            continue
        if _is_forbidden_id(sample.id):
            errors.append(
                f"Row {total} ({sample.id}): forbidden ID — contains eval-only or benchmark substring."
            )

    passed = total == expected_count and len(errors) == 0
    if total != expected_count:
        errors.append(
            f"Count mismatch: got {total} rows, expected {expected_count}."
        )

    return DatasetPreflightReport(
        dataset_type="sft",
        total_rows=total,
        expected_count=expected_count,
        passed=passed,
        errors=errors,
    )


def validate_dpo_for_training(path: Path, expected_count: int) -> DatasetPreflightReport:
    """Validate a DPO JSONL file for training readiness.

    Checks:
    - Every row is a valid DpoPair.
    - No row ID contains forbidden substrings (frozen, diagnostic, etc.).
    - Total row count matches expected_count.
    """
    errors: list[str] = []
    total = 0

    if not path.exists():
        return DatasetPreflightReport(
            dataset_type="dpo",
            total_rows=0,
            expected_count=expected_count,
            passed=False,
            errors=[f"File not found: {path}"],
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Row {total}: invalid JSON — {e}")
            continue
        try:
            pair = DpoPair(**row)
        except ValidationError as e:
            errors.append(f"Row {total}: DpoPair validation error — {e}")
            continue
        if _is_forbidden_id(pair.id):
            errors.append(
                f"Row {total} ({pair.id}): forbidden ID — contains eval-only or benchmark substring."
            )

    passed = total == expected_count and len(errors) == 0
    if total != expected_count:
        errors.append(
            f"Count mismatch: got {total} rows, expected {expected_count}."
        )

    return DatasetPreflightReport(
        dataset_type="dpo",
        total_rows=total,
        expected_count=expected_count,
        passed=passed,
        errors=errors,
    )
