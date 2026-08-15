from __future__ import annotations

import json

from homechef_booking.prompts.rules import SYSTEM_RULES
from homechef_booking.schemas.decision import FinalDecision, ToolCallDecision
from homechef_booking.schemas.runtime import BookingRuntimeInput
from homechef_booking.schemas.tools import FindChefsInput

type Message = dict[str, object]

OUTPUT_CONTRACT = {
    "format": "exactly_one_raw_json_object",
    "no_markdown": True,
    "no_prose": True,
    "allowed_actions": ["final", "tool_call"],
    "final_required_keys": list(FinalDecision.model_fields),
    "tool_call_required_keys": list(ToolCallDecision.model_fields),
    "tool_call_name": "find_chefs",
    "find_chefs_argument_keys": list(FindChefsInput.model_fields),
}

class PromptBuilder:
    def build_messages(self, runtime_input: BookingRuntimeInput) -> list[Message]:
        system_payload = {
            "output_contract": OUTPUT_CONTRACT,
            "rules": SYSTEM_RULES,
            "current_time": runtime_input.current_time,
            "current_state": runtime_input.current_state.model_dump(
                mode="json",
                exclude_none=False,
            ),
            "available_tools": [
                tool.model_dump(mode="json", exclude_none=False)
                for tool in runtime_input.available_tools
            ],
        }
        messages: list[Message] = [{"role": "system", "content": json.dumps(system_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}]
        messages.extend(message.model_dump(mode="json", exclude_none=False) for message in runtime_input.history)
        if runtime_input.user_input is not None:
            messages.append({"role": "user", "content": runtime_input.user_input})
        return messages

    def messages_to_text(self, messages: list[Message]) -> str:
        return "\n".join(json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for message in messages)
