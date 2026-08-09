from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from homechef_booking.evaluation.assertions import ASSERTION_REGISTRY, run_assertions
from homechef_booking.evaluation.critical_errors import classify_critical_errors
from homechef_booking.evaluation.effective_pass import effective_pass
from homechef_booking.evaluation.evidence import derive_tool_evidence
from homechef_booking.evaluation.results import CaseResult
from homechef_booking.evaluation.sample import load_eval_cases
from homechef_booking.evaluation.scorecard import Scorecard, aggregate_scorecard, save_scorecard
from homechef_booking.evaluation.scorers.protocol import ProtocolScorer
from homechef_booking.evaluation.scorers.task_correctness import TaskCorrectnessScorer
from homechef_booking.inference.backend import GenerationParams
from homechef_booking.inference.factory import load_backend
from homechef_booking.inference.runner import run_inference


@dataclass
class EvalConfig:
    cases_path: Path
    backend_config_path: Path
    predictions_path: Path | None = None
    scorecard_path: Path | None = None
    case_results_path: Path | None = None
    known_assertions: set[str] = field(default_factory=lambda: set(ASSERTION_REGISTRY))


@dataclass
class EvalResult:
    config: EvalConfig
    case_results: list[CaseResult]
    scorecard: Scorecard


def run_evaluation(config: EvalConfig) -> EvalResult:
    cases = load_eval_cases(config.cases_path, known_assertions=config.known_assertions)
    backend = load_backend(config.backend_config_path, predictions_path=config.predictions_path)
    params = GenerationParams()
    case_results: list[CaseResult] = []
    protocol_scorer = ProtocolScorer()
    task_scorer = TaskCorrectnessScorer()
    for case in cases:
        gen = run_inference(case.id, case.input, backend, params)
        protocol = protocol_scorer.score(case, gen)
        if protocol.passed is True:
            evidence = derive_tool_evidence(case.input)
            prediction = json.loads(gen.raw_text or "{}")
            task = task_scorer.score(case, gen, evidence)
            assertions = run_assertions(case, prediction, evidence)
            critical_tags = classify_critical_errors(case, prediction, evidence)
            eff_pass = effective_pass(True, task.score, bool(critical_tags))
        else:
            task = task_scorer.score(case, gen)
            assertions = []
            critical_tags = []
            eff_pass = False
        metric_outcomes: dict[str, bool | str] = {}
        if task.score >= 0.95:
            metric_outcomes["slot_accuracy"] = True
            metric_outcomes["tool_arguments_accuracy"] = True
            metric_outcomes["state_inheritance_accuracy"] = True
            metric_outcomes["tool_timing_accuracy"] = True
            metric_outcomes["tool_fact_grounding_accuracy"] = True
            metric_outcomes["relative_time_accuracy"] = True
            metric_outcomes["confirmation_accuracy"] = True
            metric_outcomes["dietary_constraint_recall"] = True
        for tag in critical_tags:
            if tag == "chef_fabrication":
                metric_outcomes["tool_fact_grounding_accuracy"] = False
            if tag == "dietary_constraint_loss_or_reversal":
                metric_outcomes["dietary_constraint_recall"] = False
            if tag == "unavailable_to_available":
                metric_outcomes["tool_fact_grounding_accuracy"] = False
            if tag in ("claim_booking_success", "unauthorized_booking"):
                metric_outcomes["confirmation_accuracy"] = False
            if tag == "premature_find_chefs":
                metric_outcomes["tool_timing_accuracy"] = False
            if tag == "candidate_order_semantic_mutation":
                metric_outcomes["tool_fact_grounding_accuracy"] = False
            if tag == "stale_chef_id":
                metric_outcomes["confirmation_accuracy"] = False
                metric_outcomes["state_inheritance_accuracy"] = False
        case_result = CaseResult(
            id=case.id,
            protocol_pass=protocol.passed is True,
            structured_score=task.details.get("structured_score", 0.0),
            reply_score=task.details.get("reply_score", 0.0),
            task_correctness=task.score,
            critical_error=bool(critical_tags),
            critical_error_tags=sorted(critical_tags),
            effective_pass=eff_pass,
            dimensions=[protocol, task],
            assertions=assertions,
            metric_outcomes=metric_outcomes,
        )
        case_results.append(case_result)
    scorecard = aggregate_scorecard(case_results)
    if config.case_results_path:
        config.case_results_path.parent.mkdir(parents=True, exist_ok=True)
        config.case_results_path.write_text(json.dumps([case.model_dump(mode="json", exclude_none=False) for case in case_results], ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if config.scorecard_path:
        save_scorecard(scorecard, config.scorecard_path)
    return EvalResult(config=config, case_results=case_results, scorecard=scorecard)
