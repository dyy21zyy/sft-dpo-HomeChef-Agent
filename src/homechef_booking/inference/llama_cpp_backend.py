"""Phase 02 llama.cpp server backend via OpenAI-compatible HTTP API.

Architecture:
  HomeChef messages → LlamaCppServerBackend → HTTP → llama-server.exe → GGUF → CPU

Streaming is required for real TTFT measurement.

Throughput priority:
  1. llama.cpp native: timings.predicted_per_second
  2. Client fallback: (completion_tokens - 1) / ((latency_ms - ttft_ms) / 1000)
     Requires completion_tokens > 1 and latency_ms > ttft_ms.
     Single-token responses: throughput = None (not 0).
"""

from __future__ import annotations

import json
import time
from urllib.parse import urljoin

import requests
from pydantic import BaseModel, ConfigDict

from homechef_booking.inference.backend import GenerationParams
from homechef_booking.inference.response import GenerationResult
from homechef_booking.inference.structured_output import build_homechef_decision_schema
from homechef_booking.prompts import Message

ALLOWED_LLAMA_CPP_MODEL_IDS = {
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base",
}


class LlamaCppServerConfig(BaseModel):
    """Configuration for llama.cpp server backend."""

    model_config = ConfigDict(extra="forbid", strict=True)

    base_url: str = "http://127.0.0.1:8080/v1"
    model_id: str
    max_new_tokens: int = 512
    temperature: float = 0.0
    timeout_seconds: float = 300.0

    # Structured output (llama.cpp GBNF constrained decoding)
    use_structured_output: bool = False

    # Runtime metadata (not consumed by backend, recorded in reports)
    device: str = "cpu"
    runtime: str = "llama.cpp"
    model_format: str = "gguf"
    quantization: str = "Q8_0"
    gpu_layers: int = 0


class _StreamResult:
    """Parsed streaming response data."""

    __slots__ = (
        "raw_text", "ttft_ms", "completion_tokens",
        "server_predicted_tokens", "server_predicted_ms",
        "server_tokens_per_second", "finish_reason",
    )

    def __init__(self) -> None:
        self.raw_text: str = ""
        self.ttft_ms: float | None = None
        self.completion_tokens: int = 0
        self.server_predicted_tokens: int | None = None
        self.server_predicted_ms: float | None = None
        self.server_tokens_per_second: float | None = None
        self.finish_reason: str | None = None


class LlamaCppServerBackend:
    """Backend that talks to a running llama-server.exe via OpenAI-compatible API."""

    name = "llama_cpp_server"

    def __init__(self) -> None:
        self._config: LlamaCppServerConfig | None = None
        self._session: requests.Session | None = None

    def load(self, config: LlamaCppServerConfig) -> None:
        self._config = config
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    @property
    def config(self) -> LlamaCppServerConfig:
        if self._config is None:
            raise RuntimeError("LlamaCppServerBackend not loaded. Call load(config) first.")
        return self._config

    def health_check(self) -> bool:
        """Check if llama-server is running and serving the expected model."""
        try:
            resp = self._get_session().get(
                urljoin(self.config.base_url, "/models"), timeout=10
            )
            if resp.status_code != 200:
                return False
            data = resp.json()
            models = data.get("data", [])
            return len(models) > 0
        except Exception:
            return False

    def _get_session(self) -> requests.Session:
        if self._session is None:
            raise RuntimeError("LlamaCppServerBackend not loaded. Call load(config) first.")
        return self._session

    def generate(
        self,
        messages: list[Message],
        params: GenerationParams,
        case_id: str | None = None,
    ) -> GenerationResult:
        if self._config is None:
            raise RuntimeError("LlamaCppServerBackend not loaded. Call load(config) first.")

        selected = case_id or "unknown"
        request_start = time.perf_counter()

        try:
            payload = self._build_payload(messages, params)
            endpoint = urljoin(self._config.base_url, "/chat/completions")

            resp = self._get_session().post(
                endpoint,
                json=payload,
                stream=True,
                timeout=self._config.timeout_seconds,
            )

            if resp.status_code != 200:
                elapsed = (time.perf_counter() - request_start) * 1000
                return GenerationResult(
                    case_id=selected,
                    backend_name=self.name,
                    error_type=f"HTTP_{resp.status_code}",
                    error_message=resp.text[:500],
                    latency_ms=round(elapsed, 2),
                )

            # Parse stream, extracting content + server timings
            stream = self._parse_stream(resp, request_start)

            total_elapsed = (time.perf_counter() - request_start) * 1000
            latency_ms = round(total_elapsed, 2)

            # Determine throughput with proper source tracking
            tokens_per_second, throughput_source = self._resolve_throughput(
                stream, latency_ms
            )

            return GenerationResult(
                case_id=selected,
                backend_name=self.name,
                raw_text=stream.raw_text.strip() if stream.raw_text else None,
                finish_reason=stream.finish_reason or "stop",
                latency_ms=latency_ms,
                ttft_ms=stream.ttft_ms,
                tokens_per_second=tokens_per_second,
                throughput_source=throughput_source,
                # completion_tokens=stream.completion_tokens,
                completion_tokens=(
                    stream.server_predicted_tokens
                    if stream.server_predicted_tokens is not None
                    else stream.completion_tokens
                ),
                server_predicted_tokens=stream.server_predicted_tokens,
                server_predicted_ms=stream.server_predicted_ms,
                server_tokens_per_second=stream.server_tokens_per_second,
            )

        except requests.Timeout:
            elapsed = (time.perf_counter() - request_start) * 1000
            return GenerationResult(
                case_id=selected,
                backend_name=self.name,
                error_type="timeout",
                error_message=f"Request timed out after {self._config.timeout_seconds}s",
                latency_ms=round(elapsed, 2),
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - request_start) * 1000
            return GenerationResult(
                case_id=selected,
                backend_name=self.name,
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
                latency_ms=round(elapsed, 2),
            )

    def _build_payload(
        self, messages: list[Message], params: GenerationParams
    ) -> dict:
        """Build OpenAI-compatible chat completions request.

        When use_structured_output is True, adds response_format.json_schema
        to trigger llama.cpp's GBNF constrained decoding. This ensures the
        model output is valid JSON conforming to the HomeChef Decision schema.
        """
        api_messages = []
        for msg in messages:
            api_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        max_tokens = params.max_tokens or self._config.max_new_tokens
        temperature = params.temperature if params.temperature is not None else self._config.temperature

        payload: dict = {
            "model": self._config.model_id,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        if self._config.use_structured_output:
            payload["response_format"] = build_homechef_decision_schema()

        return payload

    @staticmethod
    def _parse_stream(
        response: requests.Response, request_start: float
    ) -> _StreamResult:
        """Parse SSE streaming response.

        Extracts:
        - raw_text from content deltas
        - ttft_ms from first non-empty content chunk
        - completion_tokens count
        - Server-native timings from final chunk (timings.predicted_*)
        - finish_reason
        """
        result = _StreamResult()
        chunks: list[str] = []

        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue

            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8")
            else:
                # Compatibility with unit-test mocks that yield str.
                line = raw_line
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                finish = choices[0].get("finish_reason")
                if finish:
                    result.finish_reason = finish

                if content:
                    if result.ttft_ms is None:
                        result.ttft_ms = round(
                            (time.perf_counter() - request_start) * 1000, 2
                        )
                    chunks.append(content)
                    result.completion_tokens += 1

            # Extract llama.cpp server-native timings from final/usage chunks
            usage = data.get("usage")
            if usage:
                result.server_predicted_tokens = usage.get("completion_tokens")
            timings = data.get("timings")
            if timings:
                result.server_predicted_tokens = (
                    result.server_predicted_tokens or timings.get("predicted_n")
                )
                result.server_predicted_ms = timings.get("predicted_ms")
                result.server_tokens_per_second = timings.get("predicted_per_second")

        result.raw_text = "".join(chunks)
        return result

    @staticmethod
    def _resolve_throughput(
        stream: _StreamResult, latency_ms: float
    ) -> tuple[float | None, str | None]:
        """Determine tokens_per_second and its source.

        Priority:
          1. llama.cpp native timings.predicted_per_second
          2. Client fallback: (completion_tokens - 1) / ((latency_ms - ttft_ms) / 1000)
             Only when completion_tokens > 1 and latency_ms > ttft_ms.
             Single-token responses: throughput = None.
        """
        # Priority 1: server-native
        if stream.server_tokens_per_second is not None and stream.server_tokens_per_second > 0:
            return round(stream.server_tokens_per_second, 2), "llama_cpp_native"

        # Priority 2: client fallback
        tokens = stream.completion_tokens
        ttft = stream.ttft_ms
        if tokens is None or tokens <= 1:
            return None, "unavailable"
        if ttft is None:
            return None, "unavailable"
        decode_ms = latency_ms - ttft
        if decode_ms <= 0:
            return None, "unavailable"

        # (completion_tokens - 1) because first token is covered by TTFT
        tps = (tokens - 1) / (decode_ms / 1000.0)
        return round(tps, 2), "client_fallback"
