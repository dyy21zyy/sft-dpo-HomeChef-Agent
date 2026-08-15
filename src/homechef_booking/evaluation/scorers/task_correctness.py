from __future__ import annotations

import json

from homechef_booking.evaluation.evidence import ToolEvidence
from homechef_booking.evaluation.results import DimensionScore
from homechef_booking.evaluation.sample import EvalCase
from homechef_booking.evaluation.scorers.reply_semantics import score_reply
from homechef_booking.evaluation.scorers.semantic_slots import (
    Phase01DeterministicEmbedder,
    TextEmbedder,
    semantic_slot_f1,
)
from homechef_booking.inference.response import GenerationResult

TOOL_ARGUMENT_KEYS = ["chef_name", "service_date", "start_time", "people", "address", "cuisine", "budget_min", "budget_max", "menu", "ingredient_purchase", "dietary_constraints", "occasion"]
FINAL_FIELD_KEYS = ["booking_state.service_date", "booking_state.start_time", "booking_state.people", "booking_state.address", "booking_state.cuisine", "booking_state.budget_min", "booking_state.budget_max", "booking_state.menu", "booking_state.chef_id", "booking_state.chef_name", "booking_state.ingredient_purchase", "booking_state.dietary_constraints", "booking_state.occasion", "booking_state.confirmation", "chef_query_status", "candidate_chefs", "info_complete", "unrelated", "missing_info", "reply_type"]


class TaskCorrectnessScorer:
    name = "task_correctness"

    def __init__(self, embedder: TextEmbedder | None = None) -> None:
        self._embedder = embedder or Phase01DeterministicEmbedder()

    def score(self, case: EvalCase, generation: GenerationResult, evidence: ToolEvidence | None = None) -> DimensionScore:
        evidence = evidence or ToolEvidence()

        # Decoupled scoring: Task Correctness is independent of Protocol.
        # Score task even when protocol fails, as long as JSON is parseable.
        try:
            predicted = json.loads(generation.raw_text or "{}")
        except (json.JSONDecodeError, TypeError):
            return DimensionScore(
                name=self.name, score=0.0, passed=False,
                details={"parseable": False, "structured_score": 0.0, "reply_score": 0.0},
            )

        if not isinstance(predicted, dict):
            return DimensionScore(
                name=self.name, score=0.0, passed=False,
                details={"parseable": False, "structured_score": 0.0, "reply_score": 0.0},
            )

        expected = case.expected.model_dump(mode="json", exclude_none=False)
        if case.output_kind == "tool_call":
            checks = self._tool_checks(expected, predicted)
            reply_score = 1.0
        else:
            checks = self._final_checks(expected, predicted)
            reply_score = score_reply(predicted.get("reply"), case.reply_expectations, evidence)
        structured_score = sum(checks.values()) / len(checks)
        task_score = 0.70 * structured_score + 0.30 * reply_score
        return DimensionScore(name=self.name, score=task_score, passed=task_score >= 0.95, details={"structured_score": structured_score, "reply_score": reply_score, "structured_checks": checks})

    def _tool_checks(self, expected: dict[str, object], predicted: dict[str, object]) -> dict[str, float]:
        expected_args = expected.get("arguments", {})
        predicted_args = predicted.get("arguments", {})
        if not isinstance(predicted_args, dict):
            predicted_args = {}
        checks = {"action": _exact(predicted.get("action"), expected.get("action")), "tool_name": _exact(predicted.get("tool_name"), expected.get("tool_name"))}
        for key in TOOL_ARGUMENT_KEYS:
            checks[f"arguments.{key}"] = self._slot_score(key, expected_args.get(key), predicted_args.get(key))
        return checks

    def _final_checks(self, expected: dict[str, object], predicted: dict[str, object]) -> dict[str, float]:
        checks = {"action": _exact(predicted.get("action"), expected.get("action"))}
        for key in FINAL_FIELD_KEYS:
            field_name = key.rsplit(".", 1)[-1]
            checks[key] = self._slot_score(field_name, self._get(expected, key), self._get(predicted, key))
        return checks

    def _slot_score(self, field_name: str, expected: object, predicted: object) -> float:
        if field_name not in {"cuisine", "menu", "dietary_constraints", "occasion"}:
            return _exact(predicted, expected)
        expected_values = expected if isinstance(expected, list) else ([] if expected is None else [str(expected)])
        predicted_values = predicted if isinstance(predicted, list) else ([] if predicted is None else [str(predicted)])
        return semantic_slot_f1(list(expected_values), list(predicted_values), self._embedder)["f1"]

    def _get(self, payload: dict[str, object], key: str) -> object:
        current: object = payload
        for part in key.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current


def _exact(predicted: object, expected: object) -> float:
    return 1.0 if predicted == expected else 0.0
