from __future__ import annotations

import asyncio
import time

from .client import ClientSettings, OpenAICompatibleClient
from .metrics import summarize
from .models import BenchmarkConfig, BenchmarkRun, RequestResult
from .prometheus import fetch_metrics_snapshot
from .system_info import collect_system_info


async def run_benchmark(
    *,
    config: BenchmarkConfig,
    prompts: list[str],
    api_key: str | None = None,
) -> BenchmarkRun:
    semaphore = asyncio.Semaphore(config.concurrency)
    metrics_before = await fetch_metrics_snapshot(config.base_url)

    async with OpenAICompatibleClient(
        ClientSettings(config.base_url, api_key=api_key, timeout_s=config.timeout_s)
    ) as client:

        async def execute(index: int, prompt: str) -> RequestResult:
            if config.request_rate and config.request_rate > 0:
                await asyncio.sleep(index / config.request_rate)
            async with semaphore:
                return await client.run_chat_request(
                    request_id=index,
                    prompt=prompt,
                    model=config.model,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )

        started = time.perf_counter()
        tasks = [asyncio.create_task(execute(i, prompt)) for i, prompt in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        duration_s = time.perf_counter() - started

    metrics_after = await fetch_metrics_snapshot(config.base_url)
    return BenchmarkRun(
        config=config,
        duration_s=duration_s,
        results=results,
        summary=summarize(results, duration_s),
        system=collect_system_info(),
        metrics_before=metrics_before,
        metrics_after=metrics_after,
    )
