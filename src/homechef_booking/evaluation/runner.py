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
from homechef_booking.inference.response import GenerationResult
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
    generation_results: list[GenerationResult | None]
    scorecard: Scorecard


def run_evaluation(config: EvalConfig) -> EvalResult:
    cases = load_eval_cases(config.cases_path, known_assertions=config.known_assertions)
    backend = load_backend(config.backend_config_path, predictions_path=config.predictions_path)
    params = GenerationParams()
    case_results: list[CaseResult] = []
    generation_results: list[GenerationResult | None] = []
    protocol_scorer = ProtocolScorer()
    task_scorer = TaskCorrectnessScorer()
    for case in cases:
        gen = run_inference(case.id, case.input, backend, params)
        generation_results.append(gen)
        protocol = protocol_scorer.score(case, gen)

        # Decoupled scoring: Protocol and Task Correctness are independent.
        # Task Correctness is scored even when Protocol fails, as long as
        # the output is parseable JSON. Only truly unparseable output gets Task=0.
        try:
            evidence = derive_tool_evidence(case.input)
        except Exception:
            evidence = None

        # Try to parse raw_text for task scoring
        parseable = False
        prediction = {}
        try:
            prediction = json.loads(gen.raw_text or "")
            parseable = isinstance(prediction, dict)
        except (json.JSONDecodeError, TypeError):
            pass

        if parseable:
            task = task_scorer.score(case, gen, evidence)
            try:
                assertions = run_assertions(case, prediction, evidence)
            except Exception:
                assertions = []
            try:
                critical_tags = classify_critical_errors(case, prediction, evidence)
            except Exception:
                critical_tags = []
        else:
            # Unparseable: no evidence, scorer returns 0
            task = task_scorer.score(case, gen, None)
            assertions = []
            critical_tags = []

        # Effective Pass = Protocol PASS AND Task >= 0.95 AND no critical error
        eff_pass = effective_pass(
            protocol.passed is True,
            task.score,
            bool(critical_tags),
        )
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
        # Save enriched case results with full performance data
        enriched: list[dict] = []
        for cr, gr in zip(case_results, generation_results, strict=True):
            entry = cr.model_dump(mode="json", exclude_none=False)
            if gr is not None:
                entry["generation_result"] = {
                    "raw_text": gr.raw_text,
                    "finish_reason": gr.finish_reason,
                    "error_type": gr.error_type,
                    "error_message": gr.error_message,
                    "latency_ms": gr.latency_ms,
                    "ttft_ms": gr.ttft_ms,
                    "tokens_per_second": gr.tokens_per_second,
                    "throughput_source": gr.throughput_source,
                    "prompt_tokens": gr.prompt_tokens,
                    "completion_tokens": gr.completion_tokens,
                    "server_predicted_tokens": gr.server_predicted_tokens,
                    "server_predicted_ms": gr.server_predicted_ms,
                    "server_tokens_per_second": gr.server_tokens_per_second,
                    # Phase04 provenance: proves which trained model ran
                    # (e.g. 1.7B-SFT), not the Base. adapter provenance persists
                    # to disk, not only in backend in-memory config.
                    "base_model_id": gr.base_model_id,
                    "adapter_name_or_path": gr.adapter_name_or_path,
                    "training_stage": gr.training_stage,
                    "model_size": gr.model_size,
                    "use_structured_output": gr.use_structured_output,
                    "pref_beta": gr.pref_beta,
                }
            enriched.append(entry)
        config.case_results_path.write_text(
            json.dumps(enriched, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if config.scorecard_path:
        save_scorecard(scorecard, config.scorecard_path)
    return EvalResult(config=config, case_results=case_results, generation_results=generation_results, scorecard=scorecard)
