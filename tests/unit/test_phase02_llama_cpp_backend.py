"""Phase 02 llama.cpp server backend unit tests.

All tests use mock HTTP responses — no real GGUF or llama-server required.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from homechef_booking.inference.backend import GenerationParams
from homechef_booking.inference.llama_cpp_backend import (
    ALLOWED_LLAMA_CPP_MODEL_IDS,
    LlamaCppServerBackend,
    LlamaCppServerConfig,
)

# ── Config tests ──────────────────────────────────────────────

def test_llama_cpp_config_defaults():
    config = LlamaCppServerConfig(model_id="Qwen/Qwen3-0.6B-Base")
    assert config.base_url == "http://127.0.0.1:8080/v1"
    assert config.model_id == "Qwen/Qwen3-0.6B-Base"
    assert config.max_new_tokens == 512
    assert config.temperature == 0.0
    assert config.timeout_seconds == 300.0
    assert config.device == "cpu"
    assert config.runtime == "llama.cpp"
    assert config.model_format == "gguf"
    assert config.gpu_layers == 0


def test_llama_cpp_config_custom():
    config = LlamaCppServerConfig(
        model_id="Qwen/Qwen3-4B-Base",
        base_url="http://127.0.0.1:8080/v1",
        max_new_tokens=256,
        temperature=0.0,
        timeout_seconds=900.0,
        quantization="Q4_K_M",
    )
    assert config.quantization == "Q4_K_M"
    assert config.timeout_seconds == 900.0


def test_llama_cpp_config_accepts_all_approved_models():
    for model_id in ALLOWED_LLAMA_CPP_MODEL_IDS:
        config = LlamaCppServerConfig(model_id=model_id)
        assert config.model_id == model_id


# ── Factory creation test ─────────────────────────────────────

def test_factory_creates_llama_cpp_backend(tmp_path):
    from homechef_booking.inference.factory import load_backend

    config_path = tmp_path / "llama_cpp.yaml"
    config_path.write_text("""
backend: llama_cpp_server
base_url: http://127.0.0.1:8080/v1
model_id: Qwen/Qwen3-0.6B-Base
device: cpu
runtime: llama.cpp
model_format: gguf
quantization: Q8_0
gpu_layers: 0
max_new_tokens: 512
temperature: 0
timeout_seconds: 300
""".strip(), encoding="utf-8")

    backend = load_backend(config_path)
    assert backend.name == "llama_cpp_server"
    assert backend.config.model_id == "Qwen/Qwen3-0.6B-Base"
    assert backend.config.quantization == "Q8_0"
    assert backend.config.gpu_layers == 0


# ── Health check tests ────────────────────────────────────────

def test_health_check_healthy():
    backend = LlamaCppServerBackend()
    backend.load(LlamaCppServerConfig(model_id="Qwen/Qwen3-0.6B-Base"))

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"id": "qwen3-0.6b-base"}]}

    with patch.object(backend._session, "get", return_value=mock_response):
        assert backend.health_check() is True


def test_health_check_unhealthy_status():
    backend = LlamaCppServerBackend()
    backend.load(LlamaCppServerConfig(model_id="Qwen/Qwen3-0.6B-Base"))

    mock_response = MagicMock()
    mock_response.status_code = 500

    with patch.object(backend._session, "get", return_value=mock_response):
        assert backend.health_check() is False


def test_health_check_empty_models():
    backend = LlamaCppServerBackend()
    backend.load(LlamaCppServerConfig(model_id="Qwen/Qwen3-0.6B-Base"))

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": []}

    with patch.object(backend._session, "get", return_value=mock_response):
        assert backend.health_check() is False


def test_health_check_connection_error():
    backend = LlamaCppServerBackend()
    backend.load(LlamaCppServerConfig(model_id="Qwen/Qwen3-0.6B-Base"))

    with patch.object(backend._session, "get", side_effect=requests.ConnectionError):
        assert backend.health_check() is False


# ── SSE parsing helpers ───────────────────────────────────────

def _make_stream_chunks(chunks: list[str], timings: dict | None = None) -> list[str]:
    """Build SSE stream lines from content chunks."""
    lines = []
    for chunk in chunks:
        data = json.dumps({
            "choices": [{"delta": {"content": chunk}, "index": 0}],
            "created": int(time.time()),
            "model": "test",
        })
        lines.append(f"data: {data}")
    # Final chunk with timings
    if timings:
        final = {
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
            "timings": timings,
        }
        lines.append(f"data: {json.dumps(final)}")
    lines.append("data: [DONE]")
    return lines


def _make_usage_chunk(completion_tokens: int) -> str:
    data = json.dumps({
        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        "usage": {"completion_tokens": completion_tokens, "prompt_tokens": 100, "total_tokens": 100 + completion_tokens},
    })
    return f"data: {data}"


# ── Generate tests ────────────────────────────────────────────

def test_generate_basic_stream():
    backend = LlamaCppServerBackend()
    backend.load(LlamaCppServerConfig(model_id="Qwen/Qwen3-0.6B-Base"))

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = _make_stream_chunks(
        ['{"action":"tool_call","tool_name":"find_chefs","arguments":{', '"chef_name":null', '}}']
    )

    with patch.object(backend._session, "post", return_value=mock_response):
        result = backend.generate(
            [{"role": "user", "content": "find chefs"}],
            GenerationParams(),
            case_id="test_001",
        )

    assert result.case_id == "test_001"
    assert result.backend_name == "llama_cpp_server"
    assert result.raw_text == '{"action":"tool_call","tool_name":"find_chefs","arguments":{"chef_name":null}}'
    assert result.finish_reason == "stop"
    assert result.latency_ms is not None
    assert result.latency_ms > 0
    assert result.ttft_ms is not None
    assert result.ttft_ms > 0
    assert result.completion_tokens == 3
    assert result.error_type is None


def test_generate_throughput_server_native():
    """When server returns timings.predicted_per_second, use it."""
    backend = LlamaCppServerBackend()
    backend.load(LlamaCppServerConfig(model_id="Qwen/Qwen3-0.6B-Base"))

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = _make_stream_chunks(
        ["Hello", " world", "!"],
        timings={"predicted_n": 3, "predicted_ms": 150.0, "predicted_per_second": 20.0},
    )

    with patch.object(backend._session, "post", return_value=mock_response):
        result = backend.generate(
            [{"role": "user", "content": "say hi"}],
            GenerationParams(),
            case_id="test_002",
        )

    assert result.tokens_per_second == 20.0
    assert result.throughput_source == "llama_cpp_native"
    assert result.server_tokens_per_second == 20.0
    assert result.server_predicted_tokens == 3
    assert result.server_predicted_ms == 150.0


def test_generate_throughput_client_fallback():
    """When no server timings, use client fallback."""
    backend = LlamaCppServerBackend()
    backend.load(LlamaCppServerConfig(model_id="Qwen/Qwen3-0.6B-Base"))

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = _make_stream_chunks(["A", "B", "C"])

    with patch.object(backend._session, "post", return_value=mock_response):
        result = backend.generate(
            [{"role": "user", "content": "test"}],
            GenerationParams(),
            case_id="test_003",
        )

    assert result.completion_tokens == 3
    assert result.throughput_source == "client_fallback"
    if result.tokens_per_second is not None:
        assert result.tokens_per_second > 0


def test_generate_single_token_throughput_unavailable():
    """Single-token response: throughput = None, source = unavailable."""
    backend = LlamaCppServerBackend()
    backend.load(LlamaCppServerConfig(model_id="Qwen/Qwen3-0.6B-Base"))

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = _make_stream_chunks(["X"])

    with patch.object(backend._session, "post", return_value=mock_response):
        result = backend.generate(
            [{"role": "user", "content": "test"}],
            GenerationParams(),
            case_id="test_004",
        )

    assert result.completion_tokens == 1
    assert result.tokens_per_second is None
    assert result.throughput_source == "unavailable"


def test_generate_http_error():
    backend = LlamaCppServerBackend()
    backend.load(LlamaCppServerConfig(model_id="Qwen/Qwen3-0.6B-Base"))

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch.object(backend._session, "post", return_value=mock_response):
        result = backend.generate(
            [{"role": "user", "content": "test"}],
            GenerationParams(),
            case_id="test_005",
        )

    assert result.error_type == "HTTP_500"
    assert result.case_id == "test_005"


def test_generate_timeout():
    backend = LlamaCppServerBackend()
    backend.load(LlamaCppServerConfig(model_id="Qwen/Qwen3-0.6B-Base", timeout_seconds=0.01))

    with patch.object(backend._session, "post", side_effect=requests.Timeout):
        result = backend.generate(
            [{"role": "user", "content": "test"}],
            GenerationParams(),
            case_id="test_006",
        )

    assert result.error_type == "timeout"
    assert result.latency_ms is not None


def test_generate_exception():
    backend = LlamaCppServerBackend()
    backend.load(LlamaCppServerConfig(model_id="Qwen/Qwen3-0.6B-Base"))

    with patch.object(backend._session, "post", side_effect=RuntimeError("unexpected")):
        result = backend.generate(
            [{"role": "user", "content": "test"}],
            GenerationParams(),
            case_id="test_007",
        )

    assert result.error_type == "RuntimeError"
    assert "unexpected" in (result.error_message or "")


def test_generate_malformed_stream_no_crash():
    backend = LlamaCppServerBackend()
    backend.load(LlamaCppServerConfig(model_id="Qwen/Qwen3-0.6B-Base"))

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = [
        "data: not-json",
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}",
        "",
        "garbage line",
        "data: [DONE]",
    ]

    with patch.object(backend._session, "post", return_value=mock_response):
        result = backend.generate(
            [{"role": "user", "content": "test"}],
            GenerationParams(),
            case_id="test_008",
        )

    assert result.raw_text == "ok"
    assert result.completion_tokens == 1
    # Single token → throughput unavailable
    assert result.throughput_source == "unavailable"


def test_generate_raises_when_not_loaded():
    backend = LlamaCppServerBackend()
    with pytest.raises(RuntimeError, match="not loaded"):
        backend.generate(
            [{"role": "user", "content": "test"}],
            GenerationParams(),
        )


def test_config_property_raises_when_not_loaded():
    backend = LlamaCppServerBackend()
    with pytest.raises(RuntimeError, match="not loaded"):
        _ = backend.config


# ── Throughput resolution tests ───────────────────────────────

def test_resolve_throughput_native_priority():
    """Server native timing takes priority."""
    from homechef_booking.inference.llama_cpp_backend import _StreamResult

    stream = _StreamResult()
    stream.server_tokens_per_second = 25.0
    stream.completion_tokens = 10
    stream.ttft_ms = 100.0

    tps, source = LlamaCppServerBackend._resolve_throughput(stream, 1000.0)
    assert tps == 25.0
    assert source == "llama_cpp_native"


def test_resolve_throughput_client_fallback():
    """Client fallback: (completion_tokens - 1) / decode_seconds."""
    from homechef_booking.inference.llama_cpp_backend import _StreamResult

    stream = _StreamResult()
    stream.completion_tokens = 10
    stream.ttft_ms = 200.0

    # 10 tokens total, ttft=200ms, latency=1000ms → decode=800ms for 9 tokens
    # tps = 9 / 0.8 = 11.25
    tps, source = LlamaCppServerBackend._resolve_throughput(stream, 1000.0)
    assert tps == 11.25
    assert source == "client_fallback"


def test_resolve_throughput_single_token():
    """Single token: throughput unavailable."""
    from homechef_booking.inference.llama_cpp_backend import _StreamResult

    stream = _StreamResult()
    stream.completion_tokens = 1
    stream.ttft_ms = 500.0

    tps, source = LlamaCppServerBackend._resolve_throughput(stream, 600.0)
    assert tps is None
    assert source == "unavailable"


def test_resolve_throughput_no_ttft():
    """No TTFT: throughput unavailable."""
    from homechef_booking.inference.llama_cpp_backend import _StreamResult

    stream = _StreamResult()
    stream.completion_tokens = 10
    stream.ttft_ms = None

    tps, source = LlamaCppServerBackend._resolve_throughput(stream, 1000.0)
    assert tps is None
    assert source == "unavailable"


def test_resolve_throughput_zero_decode():
    """ttft equals latency: no decode time → unavailable."""
    from homechef_booking.inference.llama_cpp_backend import _StreamResult

    stream = _StreamResult()
    stream.completion_tokens = 5
    stream.ttft_ms = 500.0

    tps, source = LlamaCppServerBackend._resolve_throughput(stream, 500.0)
    assert tps is None
    assert source == "unavailable"


def test_resolve_throughput_native_zero():
    """Server native returns 0 tps → fall through to client fallback."""
    from homechef_booking.inference.llama_cpp_backend import _StreamResult

    stream = _StreamResult()
    stream.server_tokens_per_second = 0.0  # native but zero → skip
    stream.completion_tokens = 10
    stream.ttft_ms = 200.0

    tps, source = LlamaCppServerBackend._resolve_throughput(stream, 1000.0)
    assert tps == 11.25
    assert source == "client_fallback"


# ── Payload construction test ─────────────────────────────────

def test_build_payload_structure():
    backend = LlamaCppServerBackend()
    backend.load(LlamaCppServerConfig(model_id="Qwen/Qwen3-0.6B-Base"))

    payload = backend._build_payload(
        [{"role": "system", "content": "You are a booking agent."},
         {"role": "user", "content": "Book a chef"}],
        GenerationParams(temperature=0.0, max_tokens=512),
    )

    assert payload["model"] == "Qwen/Qwen3-0.6B-Base"
    assert payload["stream"] is True
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 512
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"


# ── Usage extraction test ─────────────────────────────────────

def test_parse_stream_extracts_usage_completion_tokens():
    backend = LlamaCppServerBackend()
    backend.load(LlamaCppServerConfig(model_id="Qwen/Qwen3-0.6B-Base"))

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = _make_stream_chunks(
        ["hello", " world"],
        timings={"predicted_n": 2, "predicted_ms": 80.0, "predicted_per_second": 25.0},
    )

    with patch.object(backend._session, "post", return_value=mock_response):
        result = backend.generate(
            [{"role": "user", "content": "hi"}],
            GenerationParams(),
            case_id="test_usage",
        )

    assert result.server_predicted_tokens == 2
    assert result.server_predicted_ms == 80.0
    assert result.server_tokens_per_second == 25.0
    assert result.throughput_source == "llama_cpp_native"


# ── Structured output schema tests ────────────────────────────

def test_structured_schema_tool_call_branch_required():
    """tool_call branch must require exactly action, tool_name, arguments."""
    from homechef_booking.inference.structured_output import build_homechef_decision_schema
    from homechef_booking.schemas.decision import ToolCallDecision

    schema = build_homechef_decision_schema()
    any_of = schema["json_schema"]["schema"]["anyOf"]

    tool_call_branch = None
    for branch in any_of:
        if branch["properties"]["action"]["const"] == "tool_call":
            tool_call_branch = branch
            break

    assert tool_call_branch is not None, "tool_call branch not found in anyOf"
    expected = list(ToolCallDecision.model_fields)
    assert tool_call_branch["required"] == expected, (
        f"tool_call required mismatch: got {tool_call_branch['required']}, expected {expected}"
    )


def test_structured_schema_final_branch_required():
    """final branch must require exactly all FinalDecision.model_fields."""
    from homechef_booking.inference.structured_output import build_homechef_decision_schema
    from homechef_booking.schemas.decision import FinalDecision

    schema = build_homechef_decision_schema()
    any_of = schema["json_schema"]["schema"]["anyOf"]

    final_branch = None
    for branch in any_of:
        if branch["properties"]["action"]["const"] == "final":
            final_branch = branch
            break

    assert final_branch is not None, "final branch not found in anyOf"
    expected = list(FinalDecision.model_fields)
    assert final_branch["required"] == expected, (
        f"final required mismatch: got {final_branch['required']}, expected {expected}"
    )


def test_structured_schema_find_chefs_input_required():
    """FindChefsInput $defs entry must require all 12 fields."""
    from homechef_booking.inference.structured_output import build_homechef_decision_schema
    from homechef_booking.schemas.tools import FindChefsInput

    schema = build_homechef_decision_schema()
    defs = schema["json_schema"]["schema"]["$defs"]

    assert "FindChefsInput" in defs, "FindChefsInput not found in $defs"
    fci = defs["FindChefsInput"]
    expected = list(FindChefsInput.model_fields)
    assert fci["required"] == expected, (
        f"FindChefsInput required mismatch: got {fci['required']}, expected {expected}"
    )
    assert len(fci["required"]) == 12, (
        f"FindChefsInput should have 12 required fields, got {len(fci['required'])}"
    )


def test_structured_schema_top_level_structure():
    """Verify response_format wrapper structure."""
    from homechef_booking.inference.structured_output import build_homechef_decision_schema

    schema = build_homechef_decision_schema()
    assert schema["type"] == "json_schema"
    js = schema["json_schema"]
    assert js["name"] == "HomeChefDecision"
    assert js["strict"] is True
    assert js["schema"]["type"] == "object"
    assert len(js["schema"]["anyOf"]) == 2


# ── Backend name test ─────────────────────────────────────────

def test_backend_name_is_llama_cpp_server():
    backend = LlamaCppServerBackend()
    assert backend.name == "llama_cpp_server"

def test_parse_stream_decodes_utf8_bytes_explicitly():
    import json
    import time
    from unittest.mock import MagicMock

    content = "请提供日期"

    event = {
        "choices": [
            {
                "delta": {"content": content},
                "finish_reason": None,
            }
        ]
    }

    final_event = {
        "choices": [
            {
                "delta": {},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "completion_tokens": 7,
        },
        "timings": {
            "predicted_n": 7,
            "predicted_ms": 100.0,
            "predicted_per_second": 70.0,
        },
    }

    response = MagicMock()
    response.iter_lines.return_value = [
        (
            "data: "
            + json.dumps(event, ensure_ascii=False)
        ).encode("utf-8"),
        b"",
        (
            "data: "
            + json.dumps(final_event, ensure_ascii=False)
        ).encode("utf-8"),
        b"",
        b"data: [DONE]",
    ]

    result = LlamaCppServerBackend._parse_stream(
        response,
        time.perf_counter(),
    )

    assert result.raw_text == "请提供日期"
    assert result.finish_reason == "stop"
    assert result.server_predicted_tokens == 7
    assert result.server_tokens_per_second == 70.0

def test_generation_result_prefers_native_server_token_count():
    backend = LlamaCppServerBackend()
    backend.load(
        LlamaCppServerConfig(
            model_id="Qwen/Qwen3-0.6B-Base"
        )
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = _make_stream_chunks(
        ["Hello"],
        timings={
            "predicted_n": 7,
            "predicted_ms": 100.0,
            "predicted_per_second": 70.0,
        },
    )

    with patch.object(
        backend._session,
        "post",
        return_value=mock_response,
    ):
        result = backend.generate(
            [{"role": "user", "content": "hi"}],
            GenerationParams(),
            case_id="native_token_count",
        )

    assert result.server_predicted_tokens == 7
    assert result.completion_tokens == 7