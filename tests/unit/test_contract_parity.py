import json
from pathlib import Path

from homechef_booking.validation.contract_validator import (
    json_schema_structural_verdict,
    pydantic_structural_verdict,
)


def fixture_paths() -> list[Path]:
    """Return fixture paths for structural parity check.

    Only final_unrelated_not_handoff is excluded because
    unrelated=true ↔ reply_type=handoff is a cross-field business
    invariant that structural schemas cannot express.
    """
    excluded = {"final_unrelated_not_handoff"}
    return [
        path
        for path in sorted(Path("tests/fixtures/contracts").glob("*/*.json"))
        if path.stem not in excluded
    ]


def test_json_schema_and_pydantic_structural_verdicts_match() -> None:
    mismatches: list[tuple[str, bool, bool]] = []
    for path in fixture_paths():
        value = json.loads(path.read_text(encoding="utf-8"))
        schema_ok = json_schema_structural_verdict(path, value)
        pydantic_ok = pydantic_structural_verdict(path, value)
        if schema_ok != pydantic_ok:
            mismatches.append((str(path), schema_ok, pydantic_ok))
    assert mismatches == []

