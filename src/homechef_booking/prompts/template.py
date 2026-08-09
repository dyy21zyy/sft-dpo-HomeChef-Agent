from __future__ import annotations

import json
from typing import TypeAlias

from homechef_booking.prompts.rules import SYSTEM_RULES
from homechef_booking.schemas.runtime import BookingRuntimeInput

Message: TypeAlias = dict[str, object]


class PromptBuilder:
    def build_messages(self, runtime_input: BookingRuntimeInput) -> list[Message]:
        system_payload = {
            "rules": SYSTEM_RULES,
            "current_time": runtime_input.current_time,
            "current_state": runtime_input.current_state.model_dump(mode="json", exclude_none=False),
            "available_tools": [tool.model_dump(mode="json", exclude_none=False) for tool in runtime_input.available_tools],
        }
        messages: list[Message] = [{"role": "system", "content": json.dumps(system_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}]
        messages.extend(message.model_dump(mode="json", exclude_none=False) for message in runtime_input.history)
        if runtime_input.user_input is not None:
            messages.append({"role": "user", "content": runtime_input.user_input})
        return messages

    def messages_to_text(self, messages: list[Message]) -> str:
        return "\n".join(json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for message in messages)
