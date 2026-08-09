"""Strict Pydantic v2 history message schemas with tool_call_id pairing."""

from __future__ import annotations

import json
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from homechef_booking.schemas.tools import parse_find_chefs_result


class UserMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["user"]
    content: str


class AssistantTextMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["assistant"]
    content: str | None


class ToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: Literal["find_chefs"]
    arguments: str


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    type: Literal["function"]
    function: ToolCallFunction


class AssistantToolCallMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["assistant"]
    content: None = None
    tool_calls: list[ToolCall] = Field(min_length=1, max_length=1)


class ToolMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["tool"]
    tool_call_id: str
    name: Literal["find_chefs"]
    content: str


HistoryMessage = Union[UserMessage, AssistantTextMessage, AssistantToolCallMessage, ToolMessage]


def _parse_history_message(msg: dict) -> BaseModel:
    """Parse a single raw dict into the correct history message type."""
    role = msg.get("role")
    if role == "user":
        return UserMessage.model_validate(msg)
    if role == "tool":
        return ToolMessage.model_validate(msg)
    if role == "assistant":
        if "tool_calls" in msg and msg.get("tool_calls"):
            return AssistantToolCallMessage.model_validate(msg)
        return AssistantTextMessage.model_validate(msg)
    raise ValueError(f"Unknown message role: {role}")


def parse_history_messages(raw_messages: list[dict]) -> list[BaseModel]:
    """Parse a list of raw dicts into typed history messages."""
    return [_parse_history_message(msg) for msg in raw_messages]


def validate_history_sequence(messages: list[BaseModel], user_input: str | None) -> list[str]:
    """Validate a history sequence for tool_call_id pairing and resolution.

    Args:
        messages: Parsed history messages.
        user_input: The current turn's user_input. None means tool result continuation.

    Returns:
        List of error strings. Empty list means valid.
    """
    errors: list[str] = []
    pending_call_ids: set[str] = set()

    for i, msg in enumerate(messages):
        if isinstance(msg, AssistantToolCallMessage):
            for tc in msg.tool_calls:
                pending_call_ids.add(tc.id)
        elif isinstance(msg, ToolMessage):
            if msg.tool_call_id not in pending_call_ids:
                errors.append(
                    f"message {i}: tool_call_id has no pending call"
                )
            else:
                pending_call_ids.discard(msg.tool_call_id)
                try:
                    parse_find_chefs_result(json.loads(msg.content))
                except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                    errors.append(f"message {i}: invalid tool result content: {exc}")

    if user_input is None:
        if pending_call_ids:
            errors.append("unresolved tool calls remain with user_input=None")
    else:
        if pending_call_ids:
            errors.append("unresolved tool calls remain with user_input != None")

    return errors


__all__ = [
    "UserMessage",
    "AssistantTextMessage",
    "AssistantToolCallMessage",
    "ToolCallFunction",
    "ToolCall",
    "ToolMessage",
    "HistoryMessage",
    "parse_history_messages",
    "validate_history_sequence",
]
