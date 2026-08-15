"""TDD: Canonical ToolSpec + missing_info + tool-call eligibility."""

from __future__ import annotations

import json
from pathlib import Path


def _load_smoke():
    lines = Path("data/raw/phase03_smoke_v0.2.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ── ToolSpec Tests ──────────────────────────────────────────

def test_tool_spec_validates_against_model():
    """Canonical ToolSpec must pass ToolSpec.model_validate."""
    from homechef_booking.data.tool_spec_factory import build_find_chefs_tool_spec
    from homechef_booking.schemas.runtime import ToolSpec

    spec = build_find_chefs_tool_spec()
    validated = ToolSpec.model_validate(spec.model_dump(mode="json"))
    assert validated.type == "function"
    assert validated.function.name == "find_chefs"


def test_tool_spec_has_12_properties():
    from homechef_booking.data.tool_spec_factory import build_find_chefs_tool_spec
    spec = build_find_chefs_tool_spec()
    props = spec.function.parameters["properties"]
    assert len(props) == 12
    assert "chef_name" in props
    assert "dietary_constraints" in props


def test_tool_spec_required_keys():
    from homechef_booking.data.tool_spec_factory import build_find_chefs_tool_spec
    from homechef_booking.schemas.runtime import _FIND_CHEFS_REQUIRED_KEYS

    spec = build_find_chefs_tool_spec()
    required = spec.function.parameters["required"]
    assert sorted(required) == sorted(_FIND_CHEFS_REQUIRED_KEYS)


def test_tool_spec_nullability():
    """Verify nullability matches canonical contract for key fields."""
    from homechef_booking.data.tool_spec_factory import build_find_chefs_tool_spec

    spec = build_find_chefs_tool_spec()
    props = spec.function.parameters["properties"]

    # Nullable: ["string", "null"]
    nullable_fields = {"chef_name", "service_date", "start_time", "people",
                       "address", "cuisine", "budget_min", "budget_max",
                       "ingredient_purchase", "occasion"}
    for field in nullable_fields:
        prop_type = props[field]["type"]
        assert "null" in prop_type, f"{field} should be nullable, got {prop_type}"

    # Non-nullable (always list)
    assert props["menu"]["type"] == "array"
    assert props["dietary_constraints"]["type"] == "array"


def test_generator_tool_spec_matches_canonical():
    """Smoke generator's available_tools must match canonical ToolSpec."""
    from homechef_booking.data.tool_spec_factory import canonical_find_chefs_tool_dict
    from homechef_booking.schemas.runtime import ToolSpec

    canonical = canonical_find_chefs_tool_dict()
    samples = _load_smoke()

    for s in samples:
        tools = s["input"].get("available_tools", [])
        for tool in tools:
            # Must be valid ToolSpec
            ToolSpec.model_validate(tool)
            # Must match canonical
            assert tool == canonical, (
                f"Sample {s['id']}: available_tools does not match canonical ToolSpec"
            )


def test_no_flat_tool_format():
    """Generator must not produce flat {name: 'find_chefs', ...} format."""
    samples = _load_smoke()
    for s in samples:
        tools = s["input"].get("available_tools", [])
        for tool in tools:
            assert "type" in tool, f"Sample {s['id']}: tool missing 'type' key"
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]


# ── missing_info Tests ──────────────────────────────────────

_CANONICAL_SLOT_ORDER = ["service_date", "start_time", "people", "address"]


def test_missing_info_canonical_order():
    """missing_info must follow canonical slot order."""
    samples = _load_smoke()
    for s in samples:
        mi = s["expected"].get("missing_info", [])
        if not mi:
            continue
        # Verify order: canonical slots come first, in order
        ordered = [x for x in _CANONICAL_SLOT_ORDER if x in mi]
        # The actual order of canonical slots in mi must match
        canonical_in_mi = [x for x in mi if x in _CANONICAL_SLOT_ORDER]
        assert canonical_in_mi == ordered, (
            f"Sample {s['id']}: missing_info order {mi} does not match canonical {ordered}"
        )


def test_missing_info_only_known_slots():
    """missing_info must only contain known canonical slot names."""
    samples = _load_smoke()
    for s in samples:
        mi = s["expected"].get("missing_info", [])
        for slot in mi:
            assert slot in _CANONICAL_SLOT_ORDER, (
                f"Sample {s['id']}: unknown slot '{slot}' in missing_info"
            )


# ── Tool-call eligibility ───────────────────────────────────

def test_tool_call_samples_have_complete_required_slots():
    """All 4 required slots must exist in tool_call arguments dict.
    Null values acceptable when scenario doesn't specify the slot (specific chef, dietary-only, etc.)."""
    samples = _load_smoke()
    for s in samples:
        if s["output_kind"] != "tool_call":
            continue
        args = s["expected"].get("arguments", {})
        for slot in _CANONICAL_SLOT_ORDER:
            assert slot in args, (
                f"Sample {s['id']}: tool_call missing required slot '{slot}'"
            )
