from vllm_inference_benchmark.prometheus import parse_metrics_text


def test_parse_selected_prometheus_metrics() -> None:
    text = """
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="demo"} 3
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="demo"} 0.42
# TYPE unrelated gauge
unrelated 9
"""
    metrics = parse_metrics_text(text)
    assert metrics["vllm:num_requests_running"] == 3.0
    assert metrics["vllm:kv_cache_usage_perc"] == 0.42
    assert "unrelated" not in metrics
