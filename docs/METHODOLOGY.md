# Benchmarking Methodology

This document defines the measurement conventions used by the project so results remain interpretable and reproducible.

## Metrics

### End-to-end latency

Elapsed wall-clock time from immediately before the client starts the HTTP request until the streaming response is complete.

### Time to First Token (TTFT)

Client-observed time from request start until the first non-empty streamed content chunk arrives. This includes client/network overhead in addition to server queueing and prefill time.

### Time per Output Token (TPOT)

Estimated as:

```text
(end_to_end_latency - TTFT) / (completion_tokens - 1)
```

The value is only calculated when completion-token usage is available and more than one completion token was generated. It is an aggregate estimate, not a decoder trace.

### Request throughput

```text
successful_requests / benchmark_duration
```

### Output token throughput

```text
total_completion_tokens / benchmark_duration
```

## Workload controls

A benchmark result should not be compared with another result unless the following are controlled or explicitly reported:

- model and model revision
- vLLM version or container tag
- GPU type and count
- tensor/pipeline/data parallel configuration
- dtype and quantization
- prompt length distribution
- requested output length
- sampling parameters
- concurrency
- request arrival rate
- prefix-caching configuration
- speculative-decoding configuration
- warm-up policy

## Warm-up

The current CLI does not silently discard warm-up requests. If warm-up is required, run a short benchmark first and exclude it from the reported experiment. This keeps the benchmark procedure explicit.

## Networking

Client-side TTFT and latency include network time. For engine-focused measurements, run the client on the same host or a controlled low-latency network. For application-focused measurements, benchmark from the location that represents the real client path.

## Prometheus snapshots

The project reads selected values from the vLLM `/metrics` endpoint before and after a run when available. These snapshots are supporting context, not full time-series monitoring. Production studies should scrape the endpoint continuously with Prometheus or an equivalent system.

## Reporting rule

Never publish a single throughput or latency number without the model, hardware, workload, and concurrency context that produced it.
