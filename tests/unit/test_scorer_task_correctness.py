"""Task 7: Task correctness scorer tests."""

import json

from homechef_booking.evaluation.evidence import ToolEvidence
from homechef_booking.evaluation.scorers.reply_semantics import score_reply
from homechef_booking.evaluation.scorers.task_correctness import TaskCorrectnessScorer
from homechef_booking.inference.response import GenerationResult


def test_reply_semantics_uses_chinese_missing_field_phrases():
    score = score_reply("请补充服务地址和人数", {"mentions_missing_info": ["address", "people"]}, ToolEvidence())
    assert score == 1.0


def test_task_correctness_scores_exact_tool_arguments(valid_tool_call_case):
    raw_text = '{"action":"tool_call","tool_name":"find_chefs","arguments":{"chef_name":null,"service_date":"2026-08-10","start_time":"18:00","people":2,"address":"上海市徐汇区","cuisine":"川菜","budget_min":500.0,"budget_max":900.0,"menu":["水煮鱼"],"ingredient_purchase":true,"dietary_constraints":["不吃花生"],"occasion":"生日"}}'
    score = TaskCorrectnessScorer().score(valid_tool_call_case, GenerationResult(case_id=valid_tool_call_case.id, backend_name="mock", raw_text=raw_text), ToolEvidence())
    assert score.score == 1.0
    assert score.details["structured_checks"]["arguments.people"] == 1.0


def test_task_correctness_uses_injected_semantic_embedder_for_dietary_slot(valid_final_case):
    class FakeEmbedder:
        def similarity(self, left: str, right: str) -> float:
            return 0.91 if {left, right} == {"不吃花生", "花生过敏"} else 0.0

    prediction = valid_final_case.expected.model_dump(mode="json", exclude_none=False)
    prediction["booking_state"]["dietary_constraints"] = ["花生过敏"]
    raw_text = json.dumps(prediction, ensure_ascii=False, sort_keys=True)
    score = TaskCorrectnessScorer(embedder=FakeEmbedder()).score(valid_final_case, GenerationResult(case_id=valid_final_case.id, backend_name="mock", raw_text=raw_text), ToolEvidence())
    assert score.details["structured_checks"]["booking_state.dietary_constraints"] == 1.0
