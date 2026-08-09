"""Task 4: Protocol scorer tests."""

from homechef_booking.evaluation.scorers.protocol import ProtocolScorer
from homechef_booking.inference.response import GenerationResult


def test_protocol_passes_structural_final_even_when_business_wrong(valid_final_case):
    raw_text = '{"action":"final","booking_state":{"service_date":"2026-08-10","start_time":"18:00","people":2,"address":"上海市徐汇区","cuisine":"川菜","budget_min":500.0,"budget_max":900.0,"menu":["水煮鱼"],"chef_id":"chef_fake","chef_name":"不存在厨师","ingredient_purchase":true,"dietary_constraints":["不吃花生"],"occasion":"生日","confirmation":true},"chef_query_status":"matched","candidate_chefs":[{"chef_id":"chef_001","chef_name":"李师傅"}],"info_complete":true,"unrelated":false,"missing_info":[],"reply_type":"booking_authorized","reply":"已确认"}'
    score = ProtocolScorer().score(valid_final_case, GenerationResult(case_id=valid_final_case.id, backend_name="mock", raw_text=raw_text))
    assert score.score == 1.0
    assert score.passed is True


def test_protocol_rejects_markdown_fence(valid_final_case):
    score = ProtocolScorer().score(valid_final_case, GenerationResult(case_id=valid_final_case.id, backend_name="mock", raw_text="```json\n{}\n```"))
    assert score.score == 0.0
    assert score.passed is False
