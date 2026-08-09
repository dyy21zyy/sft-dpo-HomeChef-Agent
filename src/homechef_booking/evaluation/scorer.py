from __future__ import annotations

from typing import Protocol

from homechef_booking.evaluation.results import DimensionScore
from homechef_booking.evaluation.sample import EvalCase
from homechef_booking.inference.response import GenerationResult


class Scorer(Protocol):
    name: str

    def score(self, case: EvalCase, generation: GenerationResult) -> DimensionScore:
        raise NotImplementedError
