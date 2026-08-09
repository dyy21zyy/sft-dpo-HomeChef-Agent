import json
from pathlib import Path

from homechef_booking.validation.contract_validator import (
    parse_model_json,
    validate_decision,
    validate_runtime_input,
)

VALID = Path("tests/fixtures/contracts/valid")
INVALID = Path("tests/fixtures/contracts/invalid")


def load_fixture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_minimal_runtime_fixture_validates() -> None:
    assert validate_runtime_input(load_fixture(VALID / "minimal_runtime_input.json")) == []


def test_final_missing_address_has_canonical_missing_info() -> None:
    runtime_input = load_fixture(VALID / "minimal_runtime_input.json")
    decision = load_fixture(VALID / "final_missing_address.json")
    assert validate_decision(decision, runtime_input) == []


def test_markdown_fenced_model_json_is_rejected() -> None:
    try:
        parse_model_json("```json\n{\"action\":\"final\"}\n```")
    except ValueError as exc:
        assert "raw JSON" in str(exc)
    else:
        raise AssertionError("expected fenced JSON rejection")


def test_candidate_order_mutation_is_rejected() -> None:
    runtime_input = load_fixture(VALID / "history_tool_continuation.json")
    decision = load_fixture(INVALID / "final_reordered_candidates.json")
    assert (
        "candidate order"
        in "\n".join(i.message for i in validate_decision(decision, runtime_input))
    )
