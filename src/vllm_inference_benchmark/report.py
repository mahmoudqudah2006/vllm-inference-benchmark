from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .models import BenchmarkRun


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(run: BenchmarkRun, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(run.to_dict(), handle, indent=2)

    rows = [result.to_dict() for result in run.results]
    with (output_dir / "requests.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "request_id",
            "success",
            "latency_s",
            "ttft_s",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "tpot_s",
            "status_code",
            "error",
            "response_chars",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = run.summary
    req = summary["requests"]
    tok = summary["tokens"]
    lat = summary["latency_s"]
    ttft = summary["ttft_s"]
    tpot = summary["tpot_s"]

    markdown = f"""# Benchmark Report

## Configuration

| Parameter | Value |
|---|---:|
| Model | `{run.config.model}` |
| Base URL | `{run.config.base_url}` |
| Requests | {run.config.num_requests} |
| Concurrency | {run.config.concurrency} |
| Request rate | {_fmt(run.config.request_rate)} |
| Max output tokens | {run.config.max_tokens} |
| Temperature | {run.config.temperature} |

## Results

| Metric | Value |
|---|---:|
| Successful requests | {req['successful']} / {req['total']} |
| Success rate | {req['success_rate'] * 100:.2f}% |
| Benchmark duration | {run.duration_s:.3f} s |
| Request throughput | {req['throughput_req_s']:.3f} req/s |
| Output token throughput | {tok['output_throughput_tok_s']:.3f} tok/s |
| Total token throughput | {tok['total_throughput_tok_s']:.3f} tok/s |
| Latency p50 | {_fmt(lat['p50'])} s |
| Latency p95 | {_fmt(lat['p95'])} s |
| Latency p99 | {_fmt(lat['p99'])} s |
| TTFT p50 | {_fmt(ttft['p50'])} s |
| TTFT p95 | {_fmt(ttft['p95'])} s |
| TTFT p99 | {_fmt(ttft['p99'])} s |
| TPOT p50 | {_fmt(tpot['p50'])} s/token |
| TPOT p95 | {_fmt(tpot['p95'])} s/token |

## Token Counts

- Prompt tokens: {tok['prompt']}
- Completion tokens: {tok['completion']}
- Total tokens: {tok['total']}

## System

```json
{json.dumps(run.system, indent=2)}
```

## Notes

TTFT is measured by the benchmark client from request start until the first streamed content chunk.
TPOT is estimated from end-to-end latency, TTFT, and the completion-token count returned by
the server.
Do not compare results across different models, hardware, quantization settings, prompt
distributions, or vLLM configurations without controlling those variables.
"""
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
