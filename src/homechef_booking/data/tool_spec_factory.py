"""Phase 03 v0.2 — Canonical ToolSpec factory (single source of truth).

Shared between Phase 03 generator, validator, and Agent Runtime.
Do NOT duplicate ToolSpec definitions in generator or tests.
"""

from __future__ import annotations

from homechef_booking.schemas.runtime import (
    _CANONICAL_FIND_CHEFS_PROPERTIES,
    _FIND_CHEFS_REQUIRED_KEYS,
    ToolFunctionSpec,
    ToolSpec,
)

_FIND_CHEFS_DESCRIPTION = (
    "Search for available chefs. "
    "Provide as many known details as possible for better matches."
)


def build_find_chefs_tool_spec() -> ToolSpec:
    """Build canonical find_chefs ToolSpec from shared contract constants.

    Returns a validated ToolSpec that is byte-for-byte identical to
    what the Agent Runtime and Phase 03 validator expect.
    """
    return ToolSpec(
        type="function",
        function=ToolFunctionSpec(
            name="find_chefs",
            description=_FIND_CHEFS_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": _CANONICAL_FIND_CHEFS_PROPERTIES,
                "required": sorted(_FIND_CHEFS_REQUIRED_KEYS),
                "additionalProperties": False,
            },
        ),
    )


def canonical_find_chefs_tool_dict() -> dict:
    """Return the canonical find_chefs ToolSpec as a JSON-serializable dict.

    Use this in generators that produce RawBookingSample JSONL output.
    """
    return build_find_chefs_tool_spec().model_dump(mode="json")
