from pathlib import Path

REQUIRED_FILES = [
    "pyproject.toml",
    ".gitignore",
    "contracts/booking_machine_contract_v1.schema.json",
    "contracts/find_chefs_v1.schema.json",
    "contracts/contract_manifest.yaml",
    "src/homechef_booking/schemas/booking.py",
    "src/homechef_booking/schemas/decision.py",
    "src/homechef_booking/schemas/history.py",
    "src/homechef_booking/schemas/runtime.py",
    "src/homechef_booking/schemas/tools.py",
    "src/homechef_booking/validation/contract_validator.py",
]


def test_phase00_required_files_exist() -> None:
    missing = [path for path in REQUIRED_FILES if not Path(path).exists()]
    assert missing == []


def test_internal_plan_is_not_inside_target_repo() -> None:
    assert not Path("docs/superpowers/plans/2026-08-08-phase-00-contract-scaffold.md").exists()


def test_no_reference_namespace_or_legacy_tool_contract() -> None:
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/homechef_booking").rglob("*.py")
    )
    forbidden = ["slot_extractor", "find_technicians", "duration_minutes", "gender_preference"]
    assert [token for token in forbidden if token in source_text] == []
