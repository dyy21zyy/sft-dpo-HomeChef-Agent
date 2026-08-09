"""Task 5: Evidence layer tests."""

from homechef_booking.evaluation.evidence import derive_tool_evidence
from homechef_booking.schemas.runtime import BookingRuntimeInput


def test_evidence_extracts_specific_available_tool_result(valid_final_case):
    runtime_payload = valid_final_case.input.model_dump(mode="json", exclude_none=False)
    runtime_payload["history"] = [
        {"role": "user", "content": "帮我找李师傅"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_001", "type": "function", "function": {"name": "find_chefs", "arguments": '{"chef_name":"李师傅","service_date":"2026-08-10","start_time":"18:00","people":2,"address":"上海市徐汇区","cuisine":"川菜","budget_min":500.0,"budget_max":900.0,"menu":["水煮鱼"],"ingredient_purchase":true,"dietary_constraints":["不吃花生"],"occasion":"生日"}'}}]},
        {"role": "tool", "tool_call_id": "call_001", "name": "find_chefs", "content": '{"mode":"specific","status":"available","chef":{"chef_id":"chef_001","chef_name":"李师傅"}}'},
    ]
    runtime_input = BookingRuntimeInput.model_validate(runtime_payload)

    evidence = derive_tool_evidence(runtime_input)

    assert evidence.latest_status == "available"
    assert evidence.requested_chef == "李师傅"
    assert evidence.verified_chef_ids == {"chef_001"}
    assert evidence.effective_candidate_order == ["chef_001"]


def test_latest_tool_result_precedence_over_current_state_order(valid_final_case):
    runtime_payload = valid_final_case.input.model_dump(mode="json", exclude_none=False)
    runtime_payload["history"] = [
        {"role": "user", "content": "帮我找厨师"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_001", "type": "function", "function": {"name": "find_chefs", "arguments": '{"chef_name":null,"service_date":"2026-08-10","start_time":"18:00","people":2,"address":"上海市徐汇区","cuisine":"川菜","budget_min":500.0,"budget_max":900.0,"menu":["水煮鱼"],"ingredient_purchase":true,"dietary_constraints":["不吃花生"],"occasion":"生日"}'}}]},
        {"role": "tool", "tool_call_id": "call_001", "name": "find_chefs", "content": '{"mode":"search","status":"matched","candidates":[{"chef_id":"chef_001","chef_name":"李师傅"},{"chef_id":"chef_002","chef_name":"王师傅"}]}'},
    ]
    runtime_payload["current_state"]["chef_query_status"] = "available"
    runtime_payload["current_state"]["candidate_chefs"] = [{"chef_id": "chef_002", "chef_name": "王师傅"}, {"chef_id": "chef_001", "chef_name": "李师傅"}]
    evidence = derive_tool_evidence(BookingRuntimeInput.model_validate(runtime_payload))
    assert evidence.latest_status == "matched"
    assert evidence.effective_candidate_order == ["chef_001", "chef_002"]
    assert evidence.verified_chef_ids == {"chef_001", "chef_002"}
