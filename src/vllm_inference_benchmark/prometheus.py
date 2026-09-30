from __future__ import annotations

from io import StringIO

import httpx
from prometheus_client.parser import text_string_to_metric_families

SELECTED_METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
}


async def fetch_metrics_snapshot(
    base_url: str,
    *,
    timeout_s: float = 10.0,
) -> dict[str, float] | None:
    """Fetch a compact snapshot from vLLM's Prometheus-compatible /metrics endpoint."""
    url = f"{base_url.rstrip('/')}/metrics"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(url)
            response.raise_for_status()
    except (httpx.HTTPError, OSError):
        return None

    values: dict[str, float] = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name in SELECTED_METRICS:
                values[sample.name] = values.get(sample.name, 0.0) + float(sample.value)
    return values


def parse_metrics_text(text: str) -> dict[str, float]:
    """Parse selected vLLM metrics from Prometheus text; useful for tests and offline analysis."""
    values: dict[str, float] = {}
    for family in text_string_to_metric_families(StringIO(text).read()):
        for sample in family.samples:
            if sample.name in SELECTED_METRICS:
                values[sample.name] = values.get(sample.name, 0.0) + float(sample.value)
    return values
