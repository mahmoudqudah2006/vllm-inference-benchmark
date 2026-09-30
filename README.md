# vLLM Inference Benchmark

A reproducible benchmarking harness for **vLLM** and other **OpenAI-compatible LLM inference servers**.

The project measures serving performance under configurable concurrency and request rates, captures reproducibility metadata, and writes machine-readable plus human-readable reports. It is designed as both an engineering tool and a research-friendly starting point for studying LLM serving behavior.

> **Status:** initial public release. The repository contains the benchmark framework; published hardware/model result sets will be added separately so measured results are never confused with examples.

## Why this project?

Modern LLM serving is not characterized by one latency number. Useful evaluation should consider several dimensions together:

- **Request throughput** — completed requests per second
- **Output token throughput** — generated tokens per second
- **End-to-end latency** — request start to completion
- **TTFT** — time to first streamed content
- **TPOT** — estimated time per output token after first token
- **Concurrency** — maximum simultaneous in-flight requests
- **Arrival rate** — optional controlled request rate
- **Server state** — selected vLLM Prometheus metrics before and after a run
- **Reproducibility** — model, workload, Python/platform, and GPU metadata

vLLM itself provides `vllm bench` commands and recommends production-oriented benchmarking tools for broader production workloads. This repository is intentionally a compact Python harness that is easy to read, modify, instrument, and extend for research experiments.

## Architecture

```mermaid
flowchart LR
    A[JSONL Workload] --> B[Async Benchmark Runner]
    B --> C[OpenAI-compatible /v1/chat/completions]
    C --> D[vLLM Server]
    D --> E[/metrics]
    C --> F[Per-request Measurements]
    E --> G[Server Metrics Snapshot]
    F --> H[Summary + CSV + JSON + Markdown]
    G --> H
```

## Features

- Async streaming requests to `/v1/chat/completions`
- Automatic model discovery through `/v1/models`
- Configurable request count, concurrency, output length, temperature, and request rate
- Client-side TTFT and end-to-end latency measurement
- TPOT estimation when server token usage is available
- Prompt/completion/total token throughput
- Selected vLLM `/metrics` snapshots
- GPU metadata capture through `nvidia-smi` when available
- JSONL workload files
- JSON, CSV, and Markdown result artifacts
- Dockerized benchmark client
- Example NVIDIA GPU vLLM deployment
- Unit tests, Ruff linting, and GitHub Actions CI

## Quick start

### 1. Start a vLLM server

Using a local vLLM installation:

```bash
vllm serve Qwen/Qwen3-0.6B
```

Or use the official NVIDIA GPU Docker image:

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B
```

The repository also includes `deploy/docker-compose.gpu.yml`.

### 2. Install the benchmark client

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,plot]'
```

### 3. Check the server

```bash
vllm-bench health --base-url http://localhost:8000
```

### 4. Run a benchmark

```bash
vllm-bench run \
  --base-url http://localhost:8000 \
  --requests 100 \
  --concurrency 8 \
  --max-tokens 128
```

The model is auto-detected from `/v1/models` unless `--model` is supplied.

To simulate a controlled arrival rate:

```bash
vllm-bench run \
  --requests 200 \
  --concurrency 16 \
  --request-rate 8
```

## Workload format

Prompts are stored as JSONL:

```json
{"prompt":"Explain KV-cache reuse in transformer inference."}
{"prompt":"Compare QPSK and 16-QAM."}
```

Use a custom workload:

```bash
vllm-bench run --prompts path/to/prompts.jsonl --requests 500 --concurrency 32
```

If the number of requests is larger than the number of prompt records, the workload cycles through the supplied prompts.

## Result artifacts

Each run writes a timestamped directory under `results/` by default:

```text
results/20260930T190000Z/
├── result.json       # full configuration, summary, system metadata, raw measurements
├── requests.csv      # one row per request
└── report.md         # human-readable benchmark report
```

Example report fields:

| Category | Metrics |
|---|---|
| Reliability | success rate, failed requests |
| Throughput | req/s, output tok/s, total tok/s |
| Latency | mean, p50, p95, p99, min, max |
| TTFT | mean, p50, p95, p99 |
| TPOT | mean, p50, p95, p99 |
| Reproducibility | model, workload config, platform, GPU metadata |

No benchmark numbers are hard-coded in this README. Real result tables should always identify the model, hardware, quantization, workload, concurrency, and software version.

## Plotting

Install the optional plotting dependency and generate a latency histogram:

```bash
pip install -e '.[plot]'
python scripts/plot_results.py results/<run>/requests.csv \
  --output results/<run>/latency_distribution.png
```

## vLLM observability

Current vLLM servers expose Prometheus-compatible production metrics at `/metrics`. The benchmark stores compact before/after snapshots of selected values when that endpoint is reachable, including running/waiting requests, KV-cache usage, and token counters.

For deeper production monitoring, connect the server metrics to Prometheus/Grafana rather than relying only on benchmark snapshots.

## Measurement notes

**TTFT:** measured from the client immediately before the HTTP request until the first non-empty streamed content chunk is received.

**TPOT:** estimated as `(end_to_end_latency - TTFT) / (completion_tokens - 1)`. This is only emitted when the server returns completion-token usage. It should not be interpreted as an exact per-token decoder trace.

**Chunk intervals:** raw inter-arrival times between streamed content chunks are retained in JSON results. A streaming chunk can contain more than one token, so chunk interval is deliberately not labeled inter-token latency.

**Request rate:** `--request-rate` uses a fixed arrival schedule. Concurrency still caps simultaneous in-flight requests.

## Reproducible benchmarking checklist

When publishing results, record at least:

1. Model and revision
2. vLLM version/container tag
3. GPU model, count, memory, and driver
4. Quantization and dtype
5. Tensor/pipeline/data-parallel settings
6. Input prompt distribution
7. Output-token limit
8. Number of requests
9. Concurrency and arrival rate
10. Prefix caching/speculative decoding settings
11. Warm-up methodology

## Relationship to vLLM's built-in benchmarks

vLLM provides its own benchmark CLI, including online serving, throughput, and latency benchmarks. Use those tools when you want direct parity with upstream vLLM benchmark methodology. Use this repository when you want a small, inspectable codebase for custom experiments, teaching, portfolio work, or research instrumentation.

## Roadmap

- [ ] Warm-up phase and warm/cold comparison
- [ ] Concurrency and request-rate sweeps
- [ ] Prefix-cache experiments
- [ ] Quantization comparisons
- [ ] Speculative-decoding experiments
- [ ] Multi-model comparison reports
- [ ] Prometheus time-series sampling during runs
- [ ] VLM/multimodal workloads
- [ ] GPU power/energy measurements
- [ ] Publication-ready figures and experiment manifests

## References

- [vLLM documentation](https://docs.vllm.ai/)
- [OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
- [vLLM benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/)

## License

MIT License — see [LICENSE](LICENSE).

---

**Author:** Mahmoud Alqudah  
Senior Software Engineer · PhD Researcher · AI/ML & Wireless Communication Engineer
