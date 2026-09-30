import pytest

from vllm_inference_benchmark.metrics import percentile, summarize
from vllm_inference_benchmark.models import RequestResult


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)


def test_summary_counts_success_and_tokens() -> None:
    results = [
        RequestResult(0, True, 1.0, 0.2, 10, 20, 30, 0.04),
        RequestResult(1, True, 2.0, 0.3, 8, 12, 20, 0.10),
        RequestResult(2, False, 0.5, error="boom"),
    ]
    summary = summarize(results, duration_s=2.0)
    assert summary["requests"]["successful"] == 2
    assert summary["requests"]["failed"] == 1
    assert summary["tokens"]["completion"] == 32
    assert summary["requests"]["throughput_req_s"] == pytest.approx(1.0)
