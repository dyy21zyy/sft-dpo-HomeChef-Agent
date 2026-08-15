"""Phase 02 performance metrics unit tests.

Tests latency/ttft/throughput aggregation logic.
"""

from homechef_booking.evaluation.benchmark_runner import (
    LatencyMetrics,
    _compute_latency_metrics,
    _compute_percentile,
)

# ── Percentile computation ────────────────────────────────────

def test_p95_single_value():
    assert _compute_percentile([100.0], 95) == 100.0


def test_p95_empty():
    assert _compute_percentile([], 95) is None


def test_p95_multiple_values():
    values = sorted([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    p95 = _compute_percentile(values, 95)
    assert p95 is not None
    assert 90.0 <= p95 <= 100.0


def test_median():
    values = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
    p50 = _compute_percentile(values, 50)
    assert p50 == 3.0


# ── Helpers ───────────────────────────────────────────────────

def _make_gen_result(latency_ms=1000.0, ttft_ms=500.0, tps=10.0,
                     completion_tokens=5, throughput_source="client_fallback", error=None):
    return {
        "generation_result": {
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
            "tokens_per_second": tps,
            "throughput_source": throughput_source,
            "completion_tokens": completion_tokens,
            "error_type": error,
        }
    }


# ── Latency metrics from case results ─────────────────────────

def test_latency_metrics_normal():
    case_results = [
        _make_gen_result(latency_ms=1000.0, ttft_ms=500.0, tps=10.0),
        _make_gen_result(latency_ms=2000.0, ttft_ms=800.0, tps=15.0),
        _make_gen_result(latency_ms=1500.0, ttft_ms=600.0, tps=12.0),
    ]
    metrics = _compute_latency_metrics(case_results, total_wall_seconds=30.0, total_cases=3)

    assert metrics.performance_sample_count == 3
    assert metrics.successful_inference_cases == 3
    assert metrics.failed_inference_cases == 0
    assert metrics.timeout_cases == 0
    assert metrics.mean_latency_ms == 1500.0
    assert metrics.median_latency_ms == 1500.0
    assert metrics.mean_ttft_ms is not None
    assert metrics.mean_tokens_per_second is not None
    assert metrics.total_wall_seconds == 30.0
    assert metrics.mean_wall_seconds_per_case == 10.0


def test_latency_metrics_with_failures():
    case_results = [
        _make_gen_result(latency_ms=1000.0, ttft_ms=500.0, tps=10.0),
        {"generation_result": None},  # missing gen result
        _make_gen_result(latency_ms=2000.0, ttft_ms=800.0, tps=15.0),
        _make_gen_result(error="timeout"),  # timeout
        _make_gen_result(error="HTTP_500"),  # http error
    ]
    metrics = _compute_latency_metrics(case_results, total_wall_seconds=60.0, total_cases=5)

    assert metrics.performance_sample_count == 2
    assert metrics.successful_inference_cases == 2
    assert metrics.failed_inference_cases == 3  # 1 None + 1 timeout + 1 http error
    assert metrics.timeout_cases == 1
    assert metrics.mean_latency_ms == 1500.0


def test_latency_metrics_all_failures():
    case_results = [
        {"generation_result": None},
        _make_gen_result(error="timeout"),
    ]
    metrics = _compute_latency_metrics(case_results, total_wall_seconds=10.0, total_cases=2)

    assert metrics.performance_sample_count == 0
    assert metrics.successful_inference_cases == 0
    assert metrics.failed_inference_cases == 2
    assert metrics.mean_latency_ms is None
    assert metrics.mean_ttft_ms is None
    assert metrics.mean_tokens_per_second is None


def test_latency_metrics_empty():
    metrics = _compute_latency_metrics([], total_wall_seconds=0.0, total_cases=0)
    assert metrics.performance_sample_count == 0
    assert metrics.successful_inference_cases == 0


def test_latency_metrics_null_performance_fields():
    """Cases where latency/ttft/tps are None should not crash."""
    case_results = [
        {"generation_result": {"latency_ms": None, "ttft_ms": None, "tokens_per_second": None}},
    ]
    metrics = _compute_latency_metrics(case_results, total_wall_seconds=5.0, total_cases=1)
    # gen result exists but has no perf → successful_inference = 1 but perf_sample = 0
    assert metrics.successful_inference_cases == 1
    assert metrics.performance_sample_count == 0
    assert metrics.mean_latency_ms is None


def test_latency_metrics_zero_values_filtered():
    """Zero latency values should be excluded from aggregates."""
    case_results = [
        _make_gen_result(latency_ms=0.0, ttft_ms=0.0, tps=0.0),
        _make_gen_result(latency_ms=1000.0, ttft_ms=500.0, tps=10.0),
    ]
    metrics = _compute_latency_metrics(case_results, total_wall_seconds=20.0, total_cases=2)
    assert metrics.successful_inference_cases == 2
    assert metrics.performance_sample_count == 1
    assert metrics.mean_latency_ms == 1000.0


def test_latency_metrics_with_throughput_unavailable():
    """Throughput = None for single-token responses."""
    case_results = [
        _make_gen_result(latency_ms=500.0, ttft_ms=400.0, tps=None,
                         completion_tokens=1, throughput_source="unavailable"),
        _make_gen_result(latency_ms=2000.0, ttft_ms=800.0, tps=15.0,
                         completion_tokens=20, throughput_source="llama_cpp_native"),
    ]
    metrics = _compute_latency_metrics(case_results, total_wall_seconds=30.0, total_cases=2)
    assert metrics.successful_inference_cases == 2
    assert metrics.performance_sample_count == 2
    # Only the 2nd case contributes to mean_tokens_per_second
    assert metrics.mean_tokens_per_second == 15.0


# ── LatencyMetrics model serialization ────────────────────────

def test_latency_metrics_serialization():
    metrics = LatencyMetrics(
        mean_latency_ms=1500.0,
        p95_latency_ms=2000.0,
        mean_ttft_ms=600.0,
        mean_tokens_per_second=12.5,
        performance_sample_count=100,
        successful_inference_cases=115,
        failed_inference_cases=5,
        timeout_cases=2,
        total_wall_seconds=300.0,
        mean_wall_seconds_per_case=3.0,
    )
    data = metrics.model_dump(mode="json")
    assert data["mean_latency_ms"] == 1500.0
    assert data["p95_latency_ms"] == 2000.0
    assert data["performance_sample_count"] == 100
    assert data["successful_inference_cases"] == 115
    assert data["failed_inference_cases"] == 5


def test_latency_metrics_all_none():
    metrics = LatencyMetrics()
    data = metrics.model_dump(mode="json")
    assert data["mean_latency_ms"] is None
    assert data["performance_sample_count"] == 0
    assert data["successful_inference_cases"] == 0
    assert data["failed_inference_cases"] == 0
