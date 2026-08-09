"""Task 3: EvalCase and load_eval_cases tests."""

from pathlib import Path

import pytest

from homechef_booking.evaluation.sample import load_eval_cases


def test_eval_case_preflight_checks_action_assertions_and_duplicates(tmp_path: Path):
    fixture = tmp_path / "cases.jsonl"
    line = (
        '{"id":"case_missing_required_slots","output_kind":"final","conversation_kind":"single_turn",'
        '"input":{"history":[{"role":"user","content":"明晚两个人"}],"current_state":{"booking_state":{'
        '"service_date":"2026-08-10","start_time":"18:00","people":2,"address":null,"cuisine":null,'
        '"budget_min":null,"budget_max":null,"menu":[],"chef_id":null,"chef_name":null,'
        '"ingredient_purchase":null,"dietary_constraints":[],"occasion":null,"confirmation":null},'
        '"chef_query_status":"not_checked","candidate_chefs":[],"awaiting_confirmation":false},'
        '"user_input":"请继续","current_time":"2026-08-09 18:00","available_tools":[]},'
        '"expected":{"action":"final","booking_state":{"service_date":"2026-08-10","start_time":"18:00",'
        '"people":2,"address":null,"cuisine":null,"budget_min":null,"budget_max":null,"menu":[],'
        '"chef_id":null,"chef_name":null,"ingredient_purchase":null,"dietary_constraints":[],'
        '"occasion":null,"confirmation":null},"chef_query_status":"not_checked","candidate_chefs":[],'
        '"info_complete":false,"unrelated":false,"missing_info":["address"],"reply_type":"ask_address",'
        '"reply":"请补充服务地址"},"assertions":["asks_missing_info"],"tags":["missing_required_slots"],'
        '"reply_expectations":{"mentions_missing_info":["address"]},"chain_id":"chain_a","step":1}'
    )
    fixture.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate eval case id"):
        load_eval_cases(fixture, known_assertions={"asks_missing_info"})


def test_eval_case_rejects_output_kind_action_mismatch(tmp_path: Path):
    fixture = tmp_path / "cases.jsonl"
    fixture.write_text(
        '{"id":"case_bad_kind","output_kind":"tool_call","conversation_kind":"single_turn","input":{"history":[],"current_state":{"booking_state":{"service_date":null,"start_time":null,"people":null,"address":null,"cuisine":null,"budget_min":null,"budget_max":null,"menu":[],"chef_id":null,"chef_name":null,"ingredient_purchase":null,"dietary_constraints":[],"occasion":null,"confirmation":null},"chef_query_status":"not_checked","candidate_chefs":[],"awaiting_confirmation":false},"user_input":"你好","current_time":"2026-08-09 18:00","available_tools":[]},"expected":{"action":"final","booking_state":{"service_date":null,"start_time":null,"people":null,"address":null,"cuisine":null,"budget_min":null,"budget_max":null,"menu":[],"chef_id":null,"chef_name":null,"ingredient_purchase":null,"dietary_constraints":[],"occasion":null,"confirmation":null},"chef_query_status":"not_checked","candidate_chefs":[],"info_complete":false,"unrelated":false,"missing_info":["service_date"],"reply_type":"ask_service_date","reply":"请提供日期"},"assertions":["asks_missing_info"],"tags":[],"reply_expectations":{}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="output_kind"):
        load_eval_cases(fixture, known_assertions={"asks_missing_info"})
