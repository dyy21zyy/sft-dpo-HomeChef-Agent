"""Task 2 & 3 Phase 03: Raw validator smoke tests."""

from pathlib import Path

from homechef_booking.data.raw_sample import parse_raw_sample_line
from homechef_booking.data.raw_validator import validate_raw_jsonl, validate_raw_sample

VALID_TOOL_LINE = '{"id":"phase03-raw-000002","dataset_version":"phase03_v0.1","contract_id":"homechef-booking-v1","source":"synthetic","scenario":"valid_search_tool_call","output_kind":"tool_call","conversation_kind":"single_turn","tags":["valid_search_tool_call"],"input":{"history":[],"current_state":{"booking_state":{"service_date":"2026-08-10","start_time":"18:00","people":2,"address":"上海市徐汇区","cuisine":"川菜","budget_min":500.0,"budget_max":900.0,"menu":["水煮鱼"],"chef_id":null,"chef_name":null,"ingredient_purchase":true,"dietary_constraints":["不吃花生"],"occasion":"生日","confirmation":null},"chef_query_status":"not_checked","candidate_chefs":[],"awaiting_confirmation":false},"user_input":"明晚六点两个人川菜不吃花生","current_time":"2026-08-09 18:00","available_tools":[{"type":"function","function":{"name":"find_chefs","description":"Find chefs","parameters":{"type":"object","additionalProperties":false,"required":["chef_name","service_date","start_time","people","address","cuisine","budget_min","budget_max","menu","ingredient_purchase","dietary_constraints","occasion"],"properties":{"chef_name":{"type":["string","null"]},"service_date":{"type":["string","null"],"pattern":"^\\\\d{4}-\\\\d{2}-\\\\d{2}$","format":"date"},"start_time":{"type":["string","null"],"pattern":"^([01]\\\\d|2[0-3]):[0-5]\\\\d$"},"people":{"type":["integer","null"]},"address":{"type":["string","null"]},"cuisine":{"type":["string","null"]},"budget_min":{"type":["number","null"]},"budget_max":{"type":["number","null"]},"menu":{"type":"array","items":{"type":"string"}},"ingredient_purchase":{"type":["boolean","null"]},"dietary_constraints":{"type":"array","items":{"type":"string"}},"occasion":{"type":["string","null"]}}}}}]},"expected":{"action":"tool_call","tool_name":"find_chefs","arguments":{"chef_name":null,"service_date":"2026-08-10","start_time":"18:00","people":2,"address":"上海市徐汇区","cuisine":"川菜","budget_min":500.0,"budget_max":900.0,"menu":["水煮鱼"],"ingredient_purchase":true,"dietary_constraints":["不吃花生"],"occasion":"生日"}},"generation":{"generator":"openai_responses","model":"gpt-5.6-sol","seed":3001,"prompt_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","generated_at":"2026-08-10T00:00:00Z"},"review":{"status":"machine_validated","reviewer":null,"notes":[]},"dpo_targets":[]}'

VALID_FINAL_LINE = '{"id":"phase03-raw-000001","dataset_version":"phase03_v0.1","contract_id":"homechef-booking-v1","source":"synthetic","scenario":"missing_required_slots","output_kind":"final","conversation_kind":"single_turn","tags":["missing_required_slots"],"input":{"history":[],"current_state":{"booking_state":{"service_date":null,"start_time":null,"people":null,"address":null,"cuisine":null,"budget_min":null,"budget_max":null,"menu":[],"chef_id":null,"chef_name":null,"ingredient_purchase":null,"dietary_constraints":[],"occasion":null,"confirmation":null},"chef_query_status":"not_checked","candidate_chefs":[],"awaiting_confirmation":false},"user_input":"想约一个家宴","current_time":"2026-08-09 18:00","available_tools":[]},"expected":{"action":"final","booking_state":{"service_date":null,"start_time":null,"people":null,"address":null,"cuisine":null,"budget_min":null,"budget_max":null,"menu":[],"chef_id":null,"chef_name":null,"ingredient_purchase":null,"dietary_constraints":[],"occasion":"家宴","confirmation":null},"chef_query_status":"not_checked","candidate_chefs":[],"info_complete":false,"unrelated":false,"missing_info":["service_date","start_time","people","address"],"reply_type":"ask_multiple_required_fields","reply":"请补充用餐日期、开始时间、人数和服务地址。"},"generation":{"generator":"openai_responses","model":"gpt-5.6-sol","seed":3001,"prompt_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","generated_at":"2026-08-10T00:00:00Z"},"review":{"status":"machine_validated","reviewer":null,"notes":[]},"dpo_targets":["H2"]}'


def test_raw_smoke_candidate_fixture_validates():
    report = validate_raw_jsonl(Path("tests/fixtures/datasets/raw_smoke_candidate.jsonl"))
    assert report.total == 2
    assert report.error_count == 0


def test_raw_invalid_candidate_fixture_reports_errors():
    report = validate_raw_jsonl(Path("tests/fixtures/datasets/raw_invalid_candidate.jsonl"))
    assert report.total == 2
    assert report.error_count == 2
    assert any("output_kind" in error.message or "contract" in error.message for error in report.errors)


def test_tool_call_sample_requires_complete_tool_spec():
    sample = parse_raw_sample_line(VALID_TOOL_LINE)
    sample.input.available_tools.clear()
    errors = validate_raw_sample(sample)
    assert any("available_tools" in error.path for error in errors)


def test_final_sample_runs_business_validator():
    sample = parse_raw_sample_line(VALID_FINAL_LINE)
    sample.expected.missing_info = []  # force business rule violation: missing_info empty but info_complete=False
    errors = validate_raw_sample(sample)
    assert any("missing_info" in error.path for error in errors) or any("info_complete" in error.path for error in errors)
