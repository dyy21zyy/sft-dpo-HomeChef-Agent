from __future__ import annotations

import json

from pydantic import ValidationError

from homechef_booking.evaluation.results import DimensionScore
from homechef_booking.evaluation.sample import EvalCase
from homechef_booking.inference.response import GenerationResult
from homechef_booking.schemas.decision import parse_decision_obj


class ProtocolScorer:
    name = "protocol"

    def score(self, case: EvalCase, generation: GenerationResult) -> DimensionScore:
        if generation.raw_text is None:
            return DimensionScore(name=self.name, score=0.0, passed=False, details={"error": generation.error_type or "missing_raw_text"})
        raw = generation.raw_text.strip()
        if raw.startswith("```") or raw.endswith("```"):
            return DimensionScore(name=self.name, score=0.0, passed=False, details={"error": "markdown_fence"})
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return DimensionScore(name=self.name, score=0.0, passed=False, details={"error": "not_object"})
            parse_decision_obj(payload)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            return DimensionScore(name=self.name, score=0.0, passed=False, details={"error": type(exc).__name__})
        return DimensionScore(name=self.name, score=1.0, passed=True, details={})
