from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .benchmark import run_benchmark
from .client import ClientSettings, OpenAICompatibleClient
from .models import BenchmarkConfig
from .report import write_report
from .workload import expand_prompts, load_prompts

app = typer.Typer(help="Benchmark vLLM and OpenAI-compatible LLM inference servers.")
console = Console()


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


@app.command()
def health(
    base_url: str = typer.Option("http://localhost:8000", help="Inference server base URL."),
    api_key: str | None = typer.Option(None, envvar="VLLM_API_KEY", help="Optional API key."),
) -> None:
    """Check connectivity and show the first model exposed by /v1/models."""

    async def check() -> tuple[bool, str, str | None]:
        async with OpenAICompatibleClient(ClientSettings(base_url, api_key=api_key)) as client:
            ok, detail = await client.health()
            if not ok:
                return False, detail, None
            return True, detail, await client.resolve_model()

    ok, detail, model = _run(check())
    if ok:
        console.print(f"[green]Healthy[/green] — {detail} — model: [bold]{model}[/bold]")
    else:
        console.print(f"[red]Unreachable[/red] — {detail}")
        raise typer.Exit(code=1)


@app.command("run")
def run_command(
    base_url: str = typer.Option("http://localhost:8000", help="Inference server base URL."),
    model: str | None = typer.Option(None, help="Model name; auto-detected when omitted."),
    prompts: Path = typer.Option(Path("data/prompts.jsonl"), exists=True, readable=True),
    requests: int = typer.Option(50, min=1, help="Number of requests to send."),
    concurrency: int = typer.Option(4, min=1, help="Maximum in-flight requests."),
    request_rate: float | None = typer.Option(
        None, min=0.001, help="Optional fixed request arrival rate in requests/second."
    ),
    max_tokens: int = typer.Option(128, min=1, help="Maximum output tokens per request."),
    temperature: float = typer.Option(0.0, min=0.0, help="Sampling temperature."),
    timeout: float = typer.Option(180.0, min=1.0, help="Per-request timeout in seconds."),
    output_dir: Path | None = typer.Option(None, help="Directory for JSON/CSV/Markdown results."),
    api_key: str | None = typer.Option(None, envvar="VLLM_API_KEY", help="Optional API key."),
) -> None:
    """Run a streaming chat-completions benchmark."""

    async def resolve_model_name() -> str:
        if model:
            return model
        async with OpenAICompatibleClient(
            ClientSettings(base_url, api_key=api_key, timeout_s=timeout)
        ) as client:
            return await client.resolve_model()

    model_name = _run(resolve_model_name())
    prompt_list = expand_prompts(load_prompts(prompts), requests)
    config = BenchmarkConfig(
        base_url=base_url,
        model=model_name,
        num_requests=requests,
        concurrency=concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
        request_rate=request_rate,
        timeout_s=timeout,
    )

    console.print(
        f"Running [bold]{requests}[/bold] requests against [bold]{model_name}[/bold] "
        f"with concurrency [bold]{concurrency}[/bold]..."
    )
    benchmark_run = _run(run_benchmark(config=config, prompts=prompt_list, api_key=api_key))

    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path("results") / timestamp
    write_report(benchmark_run, output_dir)

    summary = benchmark_run.summary
    table = Table(title="Benchmark Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Success", f"{summary['requests']['successful']}/{summary['requests']['total']}")
    table.add_row("Request throughput", f"{summary['requests']['throughput_req_s']:.3f} req/s")
    table.add_row("Output throughput", f"{summary['tokens']['output_throughput_tok_s']:.3f} tok/s")
    table.add_row("Latency p50", f"{summary['latency_s']['p50'] or 0:.4f} s")
    table.add_row("Latency p95", f"{summary['latency_s']['p95'] or 0:.4f} s")
    table.add_row("TTFT p50", f"{summary['ttft_s']['p50'] or 0:.4f} s")
    table.add_row("TTFT p95", f"{summary['ttft_s']['p95'] or 0:.4f} s")
    console.print(table)
    console.print(f"Results written to [bold]{output_dir}[/bold]")


if __name__ == "__main__":
    app()
