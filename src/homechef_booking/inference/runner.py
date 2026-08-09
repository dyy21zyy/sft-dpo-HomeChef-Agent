from __future__ import annotations

from homechef_booking.inference.backend import Backend, GenerationParams
from homechef_booking.inference.response import GenerationResult
from homechef_booking.prompts import PromptBuilder
from homechef_booking.schemas.runtime import BookingRuntimeInput


def run_inference(case_id: str, runtime_input: BookingRuntimeInput, backend: Backend, params: GenerationParams) -> GenerationResult:
    messages = PromptBuilder().build_messages(runtime_input)
    return backend.generate(messages, params, case_id=case_id)
