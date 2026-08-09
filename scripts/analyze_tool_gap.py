"""Analyze frozen spec tool definition vs canonical, find all occurrences."""

import json
from pathlib import Path

FROZEN = Path("../.plans/sft-dpo-HomeChef-Agent/phase01_frozen_fixture_spec.json")

spec = json.loads(FROZEN.read_text(encoding="utf-8"))

# Canonical tool properties from Phase 00 runtime.py
CANONICAL = {
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

# Check top-level tool_spec
top_tool = spec.get("tool_spec", {})
if top_tool:
    params = top_tool.get("function", {}).get("parameters", {})
    if params:
        props = params.get("properties", {})
        for key in sorted(props):
            canon = CANONICAL.get(key, {})
            if props[key] != canon:
                print(f"TOP tool_spec.{key}: fixture={json.dumps(props[key], ensure_ascii=False)}")
                print(f"                   canon={json.dumps(canon, ensure_ascii=False)}")
                print()

# Check all cases
for case in spec.get("cases", []):
    tools = case.get("input", {}).get("available_tools", [])
    for tool in tools:
        params = tool.get("function", {}).get("parameters", {})
        if params:
            props = params.get("properties", {})
            for key in sorted(props):
                canon = CANONICAL.get(key, {})
                if props[key] != canon:
                    print(f"Case {case['id']} tool.{key}: fixture={json.dumps(props[key], ensure_ascii=False)}")
                    print(f"                     canon={json.dumps(canon, ensure_ascii=False)}")
                    print()
