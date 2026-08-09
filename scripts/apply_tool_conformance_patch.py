"""Apply CONTRACT CONFORMANCE PATCH to phase01_frozen_fixture_spec.json.

Replaces all find_chefs tool property definitions with Phase 00 canonical format.
Does NOT modify: case ids, case count, oracle, mock_prediction, expected business content.
"""

import json
from pathlib import Path

FROZEN = Path("../.plans/sft-dpo-HomeChef-Agent/phase01_frozen_fixture_spec.json")
OUT = FROZEN  # overwrite in place

# Phase 00 canonical FindChefs property definitions (from runtime.py _CANONICAL_FIND_CHEFS_PROPERTIES)
CANONICAL_PROPS = {
    "chef_name": {"type": ["string", "null"]},
    "service_date": {"type": ["string", "null"], "pattern": r"^\d{4}-\d{2}-\d{2}$", "format": "date"},
    "start_time": {"type": ["string", "null"], "pattern": r"^([01]\d|2[0-3]):[0-5]\d$"},
    "people": {"type": ["integer", "null"]},
    "address": {"type": ["string", "null"]},
    "cuisine": {"type": ["string", "null"]},
    "budget_min": {"type": ["number", "null"]},
    "budget_max": {"type": ["number", "null"]},
    "menu": {"type": "array", "items": {"type": "string"}},
    "ingredient_purchase": {"type": ["boolean", "null"]},
    "dietary_constraints": {"type": "array", "items": {"type": "string"}},
    "occasion": {"type": ["string", "null"]},
}

spec = json.loads(FROZEN.read_text(encoding="utf-8"))

patches_applied = 0


def patch_tool_properties(props: dict) -> int:
    """Replace mismatched property definitions with canonical ones. Returns count of patches."""
    count = 0
    for key in list(props):
        if key in CANONICAL_PROPS and props[key] != CANONICAL_PROPS[key]:
            print(f"  PATCH {key}: {json.dumps(props[key], ensure_ascii=False)} -> {json.dumps(CANONICAL_PROPS[key], ensure_ascii=False)}")
            props[key] = CANONICAL_PROPS[key]
            count += 1
    return count


# Patch top-level tool_spec
if "tool_spec" in spec:
    params = spec["tool_spec"].get("function", {}).get("parameters", {})
    if params and "properties" in params:
        print(f"Patching top-level tool_spec:")
        patches_applied += patch_tool_properties(params["properties"])

# Patch all cases
for case in spec.get("cases", []):
    tools = case.get("input", {}).get("available_tools", [])
    for tool in tools:
        params = tool.get("function", {}).get("parameters", {})
        if params and "properties" in params:
            print(f"Patching case {case['id']}:")
            patches_applied += patch_tool_properties(params["properties"])

# Write back
OUT.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\nTotal patches applied: {patches_applied}")
print(f"Output: {OUT.resolve()}")
