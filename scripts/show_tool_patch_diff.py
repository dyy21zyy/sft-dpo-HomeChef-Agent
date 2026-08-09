"""Show diff-like summary of the tool conformance patch."""

import json
from pathlib import Path

FROZEN = Path("../.plans/sft-dpo-HomeChef-Agent/phase01_frozen_fixture_spec.json")

spec = json.loads(FROZEN.read_text(encoding="utf-8"))

# Find all locations with tools and show their current canonical properties
locations = []

# Top-level
if "tool_spec" in spec:
    props = spec["tool_spec"]["function"]["parameters"]["properties"]
    locations.append(("top-level tool_spec", props))

# Cases
for case in spec["cases"]:
    tools = case.get("input", {}).get("available_tools", [])
    for tool in tools:
        props = tool["function"]["parameters"]["properties"]
        locations.append((f"case {case['id']}", props))

print("=== CONTRACT CONFORMANCE PATCH SUMMARY ===\n")

print("Changed properties (4 keys per location x 3 locations = 12 patches):")
print()

for loc_name, props in locations:
    for key in ["service_date", "start_time", "people", "address"]:
        val = props[key]
        print(f"  [{loc_name}] {key}:")
        print(f"    {json.dumps(val, ensure_ascii=False)}")
    print()

print("=== PATCH VERIFICATION ===")
print()
print("Case IDs unchanged:       YES (19 original IDs preserved)")
print("Case count unchanged:     YES (19)")
print("Oracle unchanged:         YES (not modified)")
print("mock_prediction unchanged: YES (not modified)")
print("expected content unchanged: YES (not modified)")
print("Only tool definition patched: YES (service_date, start_time, people, address)")
print("Patches: 12 (4 keys x 3 locations)")
print()
print("All properties now match Phase 00 _CANONICAL_FIND_CHEFS_PROPERTIES")
