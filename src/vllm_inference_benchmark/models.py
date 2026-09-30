from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class BenchmarkConfig:
    base_url: str
    model: str
    num_requests: int
    concurrency: int
    max_tokens: int
    temperature: float
    request_rate: float | None = None
    timeout_s: float = 180.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RequestResult:
    request_id: int
    success: bool
    latency_s: float
    ttft_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    tpot_s: float | None = None
    status_code: int | None = None
    error: str | None = None
    response_chars: int = 0
    chunk_intervals_s: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkRun:
    config: BenchmarkConfig
    duration_s: float
    results: list[RequestResult]
    summary: dict[str, Any]
    system: dict[str, Any]
    metrics_before: dict[str, float] | None = None
    metrics_after: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "duration_s": self.duration_s,
            "summary": self.summary,
            "system": self.system,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "results": [r.to_dict() for r in self.results],
        }
