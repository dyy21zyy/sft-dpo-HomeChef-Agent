from __future__ import annotations

from homechef_booking.inference.backend import GenerationParams
from homechef_booking.inference.response import GenerationResult
from homechef_booking.prompts import Message


class MockBackend:
    name = "mock"

    def __init__(self, predictions: dict[str, dict[str, str]]) -> None:
        self._predictions = predictions
        self.seen_message_counts: dict[str, int] = {}

    def generate(self, messages: list[Message], params: GenerationParams, case_id: str | None = None) -> GenerationResult:
        selected = case_id or "unknown"
        self.seen_message_counts[selected] = len(messages)
        record = self._predictions.get(selected)
        if record is None:
            return GenerationResult(case_id=selected, backend_name=self.name, error_type="missing_mock_prediction", error_message=f"No mock prediction configured for {selected}")
        if "raise" in record:
            raise RuntimeError(record["raise"])
        return GenerationResult(case_id=selected, backend_name=self.name, raw_text=record.get("raw_text"), finish_reason="stop")
