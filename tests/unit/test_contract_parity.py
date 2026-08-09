import json
from pathlib import Path

from homechef_booking.validation.contract_validator import json_schema_verdict, pydantic_verdict


def fixture_paths() -> list[Path]:
    """Return fixture paths for parity check.

    Only cross-field business invariants that JSON Schema cannot express
    are excluded from structural parity. Date and time validation are
    included because JSON Schema uses format:date with FormatChecker
    and pattern matching respectively.
    """
    excluded = {"final_unrelated_not_handoff"}
    return [
        path
        for path in sorted(Path("tests/fixtures/contracts").glob("*/*.json"))
        if path.stem not in excluded
    ]


def test_json_schema_and_pydantic_verdicts_match_for_contract_fixtures() -> None:
    mismatches: list[tuple[str, bool, bool]] = []
    for path in fixture_paths():
        value = json.loads(path.read_text(encoding="utf-8"))
        schema_ok = json_schema_verdict(path, value)
        pydantic_ok = pydantic_verdict(path, value)
        if schema_ok != pydantic_ok:
            mismatches.append((str(path), schema_ok, pydantic_ok))
    assert mismatches == []
