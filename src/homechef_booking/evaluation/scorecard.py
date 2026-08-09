from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from homechef_booking.evaluation.results import CaseResult


class AggregateMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    total_cases: int
    protocol_pass_rate: float
    mean_structured_score: float
    mean_reply_score: float
    mean_task_correctness: float
    critical_error_rate: float
    effective_pass_rate: float
    metrics: dict[str, float | str] = Field(default_factory=dict)


class Scorecard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    phase: str = "01"
    metrics: AggregateMetrics
    dimension_pass_rates: dict[str, float] = Field(default_factory=dict)
    assertion_pass_rates: dict[str, float] = Field(default_factory=dict)
    critical_error_distribution: dict[str, int] = Field(default_factory=dict)


def aggregate_scorecard(case_results: list[CaseResult], phase: str = "01") -> Scorecard:
    total = len(case_results) or 1
    protocol_pass_count = sum(1 for case in case_results if case.protocol_pass)
    critical_error_count = sum(1 for case in case_results if case.critical_error)
    effective_pass_count = sum(1 for case in case_results if case.effective_pass)
    metric_counts: dict[str, int] = defaultdict(int)
    for case in case_results:
        for key, value in case.metric_outcomes.items():
            if value is True:
                metric_counts[key] += 1
    metrics: dict[str, float | str] = {}
    for key, count in metric_counts.items():
        metrics[key] = round(count / total, 4)
    dimension_counts: dict[str, int] = defaultdict(int)
    dimension_totals: dict[str, int] = defaultdict(int)
    for case in case_results:
        for dimension in case.dimensions:
            dimension_totals[dimension.name] += 1
            if dimension.passed is True:
                dimension_counts[dimension.name] += 1
    dimension_pass_rates = {name: round(dimension_counts[name] / max(dimension_totals[name], 1), 4) for name in sorted(dimension_totals)}
    assertion_counts: dict[str, int] = defaultdict(int)
    assertion_totals: dict[str, int] = defaultdict(int)
    for case in case_results:
        for assertion in case.assertions:
            assertion_totals[assertion.name] += 1
            if assertion.passed:
                assertion_counts[assertion.name] += 1
    assertion_pass_rates = {name: round(assertion_counts[name] / max(assertion_totals[name], 1), 4) for name in sorted(assertion_totals)}
    error_dist: dict[str, int] = defaultdict(int)
    for case in case_results:
        for tag in case.critical_error_tags:
            error_dist[tag] += 1
    metrics.update({
        "tool_arguments_accuracy": metrics.get("tool_arguments_accuracy", "not_available"),
    })
    aggregate = AggregateMetrics(
        total_cases=total,
        protocol_pass_rate=round(protocol_pass_count / total, 4),
        mean_structured_score=round(sum(case.structured_score for case in case_results) / total, 4),
        mean_reply_score=round(sum(case.reply_score for case in case_results) / total, 4),
        mean_task_correctness=round(sum(case.task_correctness for case in case_results) / total, 4),
        critical_error_rate=round(critical_error_count / total, 4),
        effective_pass_rate=round(effective_pass_count / total, 4),
        metrics=dict(sorted(metrics.items())),
    )
    return Scorecard(phase=phase, metrics=aggregate, dimension_pass_rates=dimension_pass_rates, assertion_pass_rates=assertion_pass_rates, critical_error_distribution=dict(sorted(error_dist.items())))


def save_scorecard(scorecard: Scorecard, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scorecard.model_dump(mode="json", exclude_none=False), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
