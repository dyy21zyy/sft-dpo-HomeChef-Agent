import json
from pathlib import Path

from homechef_booking.validation.contract_validator import json_schema_verdict, pydantic_verdict


def fixture_paths() -> list[Path]:
    """Return fixture paths for parity check.

    Fixtures that test semantic contract rules (calendar dates, clock times,
    unrelated/handoff consistency, tool_call_id pairing) are excluded because
    JSON Schema pattern matching and structural validation cannot express
    these rules. They are tested separately in their respective test files.
    """
    excluded = {
        "booking_slot_bad_date",
        "booking_slot_bad_time_2400",
        "booking_slot_bad_time_1860",
        "final_unrelated_not_handoff",
        "history_tool_continuation",
    }
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
