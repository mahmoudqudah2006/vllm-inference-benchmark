from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from .models import RequestResult


def percentile(values: Iterable[float], percentile_value: float) -> float | None:
    data = sorted(values)
    if not data:
        return None
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between 0 and 100")
    if len(data) == 1:
        return data[0]

    rank = (len(data) - 1) * (percentile_value / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return data[lower]
    weight = rank - lower
    return data[lower] * (1 - weight) + data[upper] * weight


def _stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize(results: list[RequestResult], duration_s: float) -> dict[str, Any]:
    successful = [result for result in results if result.success]
    latencies = [result.latency_s for result in successful]
    ttfts = [result.ttft_s for result in successful if result.ttft_s is not None]
    tpots = [result.tpot_s for result in successful if result.tpot_s is not None]

    prompt_tokens = sum(result.prompt_tokens or 0 for result in successful)
    completion_tokens = sum(result.completion_tokens or 0 for result in successful)
    total_tokens = sum(result.total_tokens or 0 for result in successful)

    return {
        "requests": {
            "total": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "success_rate": len(successful) / len(results) if results else 0.0,
            "throughput_req_s": len(successful) / duration_s if duration_s > 0 else 0.0,
        },
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
            "output_throughput_tok_s": completion_tokens / duration_s if duration_s > 0 else 0.0,
            "total_throughput_tok_s": total_tokens / duration_s if duration_s > 0 else 0.0,
        },
        "latency_s": _stats(latencies),
        "ttft_s": _stats(ttfts),
        "tpot_s": _stats(tpots),
    }
