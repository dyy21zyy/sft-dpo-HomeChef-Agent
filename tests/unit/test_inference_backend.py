"""Task 2: Backend interface and MockBackend tests."""

import json
from pathlib import Path

from homechef_booking.inference.backend import GenerationParams
from homechef_booking.inference.factory import load_backend


def test_mock_backend_returns_valid_12_key_tool_call(tmp_path: Path):
    raw_text = (
        '{"action":"tool_call","tool_name":"find_chefs","arguments":{'
        '"chef_name":null,"service_date":"2026-08-10","start_time":"18:00","people":2,'
        '"address":"上海市徐汇区","cuisine":"川菜","budget_min":500.0,"budget_max":900.0,'
        '"menu":["水煮鱼"],"ingredient_purchase":true,"dietary_constraints":["不吃花生"],"occasion":"生日"}}'
    )
    config = tmp_path / "mock.yaml"
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps({"case_tool": {"raw_text": raw_text}}, ensure_ascii=False), encoding="utf-8")
    config.write_text(f"backend: mock\npredictions_path: {predictions}\n", encoding="utf-8")

    backend = load_backend(config)
    result = backend.generate([{"role": "user", "content": "prompt"}], GenerationParams(), case_id="case_tool")

    assert result.case_id == "case_tool"
    assert result.backend_name == "mock"
    assert result.raw_text == raw_text
    assert result.ttft_ms is None
    assert result.tokens_per_second is None
    assert result.rss_mb is None
    assert result.model_size_bytes is None
