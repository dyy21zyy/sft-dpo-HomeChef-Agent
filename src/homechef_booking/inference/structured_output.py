"""Build llama.cpp `response_format.json_schema` from HomeChef Pydantic schemas.

Architecture:
  Pydantic ToolCallDecision / FinalDecision models
    → pydantic.model_json_schema()
    → OpenAI-compatible json_schema dict (with merged $defs)
    → inserted into /v1/chat/completions payload

This triggers llama.cpp's built-in constrained decoding (GBNF grammar under the hood).
The raw_text returned by the backend is still a JSON string — no change to evaluation pipeline.

required patching:
  Pydantic model_json_schema() omits fields with default values from "required".
  For llama.cpp GBNF, we need ALL model fields in "required" (strict mode).
  We patch required AFTER the full anyOf + $defs merge using list(model_fields).
"""

from __future__ import annotations

import copy

from homechef_booking.schemas.decision import FinalDecision, ToolCallDecision
from homechef_booking.schemas.tools import FindChefsInput


def build_canonical_decision_json_schema() -> dict:
    """Build the CANONICAL Decision union JSON Schema dict (plain JSON Schema).

    This is the single source of truth for the HomeChef Decision output
    contract, derived directly from the Pydantic ``ToolCallDecision`` /
    ``FinalDecision`` models (no hand-written copy).

    It encodes the discriminated union {tool_call, final}: the model must
    output exactly one of the two alternatives.

    ``required`` is patched post-merge using ``list(model_fields)`` so every
    model field is required (strict mode) — this satisfies both llama.cpp GBNF
    and LM Format Enforcer constrained decoding.

    Returns:
        A plain JSON Schema dict (top-level ``anyOf`` union with merged
        ``$defs``), suitable for LM Format Enforcer's ``JsonSchemaParser``.
    """
    # Deep-copy to avoid mutating Pydantic's cached schema
    tool_call_schema = copy.deepcopy(ToolCallDecision.model_json_schema())
    final_schema = copy.deepcopy(FinalDecision.model_json_schema())

    # Collect $defs from both schemas and merge them at the top level
    merged_defs: dict = {}
    for schema in (tool_call_schema, final_schema):
        sub_defs = schema.pop("$defs", {})
        for name, defn in sub_defs.items():
            if name not in merged_defs:
                merged_defs[name] = defn

    # Strip Pydantic metadata from both sub-schemas and merged defs
    for schema in (tool_call_schema, final_schema):
        _strip_pydantic_meta(schema)
    for defn in merged_defs.values():
        if isinstance(defn, dict):
            _strip_pydantic_meta(defn)

    # Build the top-level canonical schema
    top_schema: dict = {
        "type": "object",
        "anyOf": [
            tool_call_schema,
            final_schema,
        ],
        "$defs": merged_defs,
    }

    # ── Patch required after full merge ──────────────────────
    _patch_required(top_schema)

    return top_schema


def build_homechef_decision_schema() -> dict:
    """Build the full Decision union JSON schema for llama.cpp constrained decoding.

    llama.cpp's response_format supports a top-level `anyOf` schema that
    encodes the discriminated union {tool_call, final}. The model must
    output exactly one of the two alternatives.

    Returns:
        OpenAI-compatible json_schema dict suitable for response_format.
        The inner ``schema`` is the canonical schema from
        ``build_canonical_decision_json_schema()`` (single source of truth).
    """
    top_schema = build_canonical_decision_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "HomeChefDecision",
            "strict": True,
            "schema": top_schema,
        },
    }


def _patch_required(schema: dict) -> None:
    """Patch "required" arrays using Pydantic model_fields.

    Pydantic model_json_schema() only puts non-default fields in "required".
    For llama.cpp GBNF strict mode, all model fields must be required.
    This runs AFTER the full anyOf + $defs merge.

    Branch identification uses branch["properties"]["action"]["const"].
    """
    for branch in schema.get("anyOf", []):
        action_const = branch.get("properties", {}).get("action", {}).get("const")
        if action_const == "tool_call":
            branch["required"] = list(ToolCallDecision.model_fields)
        elif action_const == "final":
            branch["required"] = list(FinalDecision.model_fields)

    # Patch FindChefsInput in $defs
    fci = schema.get("$defs", {}).get("FindChefsInput")
    if fci is not None:
        fci["required"] = list(FindChefsInput.model_fields)


def _strip_pydantic_meta(schema: dict) -> None:
    """Remove Pydantic-generated metadata keys that llama.cpp does not understand.

    Kept:
      - type, properties, required, additionalProperties, anyOf, const, default,
        items, enum, $ref, $defs, allOf, oneOf
    Removed:
      - title, description (top-level)
      - enumNames (llama.cpp doesn't support it)
      - any property-level title/description that might cause validation issues
    """
    schema.pop("title", None)
    schema.pop("description", None)
    schema.setdefault("additionalProperties", False)

    for prop in schema.get("properties", {}).values():
        if isinstance(prop, dict):
            prop.pop("title", None)
            prop.pop("description", None)
            prop.pop("enumNames", None)
            # Recurse into nested objects/arrays
            if prop.get("type") == "object":
                _strip_pydantic_meta(prop)
            if prop.get("type") == "array":
                items = prop.get("items", {})
                if isinstance(items, dict):
                    _strip_pydantic_meta(items)
