from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

from .models import RequestResult


@dataclass(slots=True)
class ClientSettings:
    base_url: str
    api_key: str | None = None
    timeout_s: float = 180.0


class OpenAICompatibleClient:
    def __init__(self, settings: ClientSettings) -> None:
        self.settings = settings
        headers = {"Content-Type": "application/json"}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        self._client = httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(settings.timeout_s),
        )

    async def __aenter__(self) -> OpenAICompatibleClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    async def resolve_model(self) -> str:
        response = await self._client.get("/v1/models")
        response.raise_for_status()
        payload = response.json()
        models = payload.get("data", [])
        if not models:
            raise RuntimeError("The server returned no models from /v1/models.")
        model = models[0].get("id")
        if not model:
            raise RuntimeError("The first /v1/models entry does not contain an id.")
        return str(model)

    async def health(self) -> tuple[bool, str]:
        try:
            response = await self._client.get("/v1/models")
            response.raise_for_status()
            return True, f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            return False, str(exc)

    async def run_chat_request(
        self,
        *,
        request_id: int,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> RequestResult:
        started = time.perf_counter()
        first_token_at: float | None = None
        last_content_at: float | None = None
        chunk_intervals: list[float] = []
        response_chars = 0
        status_code: int | None = None
        usage: dict[str, int] = {}

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as response:
                status_code = response.status_code
                response.raise_for_status()

                async for raw_line in response.aiter_lines():
                    if not raw_line.startswith("data:"):
                        continue
                    data = raw_line[5:].strip()
                    if not data or data == "[DONE]":
                        continue

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    event_usage = event.get("usage")
                    if isinstance(event_usage, dict):
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                            value = event_usage.get(key)
                            if isinstance(value, int):
                                usage[key] = value

                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if not isinstance(content, str) or not content:
                        continue

                    now = time.perf_counter()
                    response_chars += len(content)
                    if first_token_at is None:
                        first_token_at = now
                    if last_content_at is not None:
                        chunk_intervals.append(now - last_content_at)
                    last_content_at = now

            finished = time.perf_counter()
            latency_s = finished - started
            ttft_s = (first_token_at - started) if first_token_at is not None else None
            completion_tokens = usage.get("completion_tokens")
            tpot_s: float | None = None
            if ttft_s is not None and completion_tokens is not None and completion_tokens > 1:
                tpot_s = max(latency_s - ttft_s, 0.0) / (completion_tokens - 1)

            return RequestResult(
                request_id=request_id,
                success=True,
                latency_s=latency_s,
                ttft_s=ttft_s,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=completion_tokens,
                total_tokens=usage.get("total_tokens"),
                tpot_s=tpot_s,
                status_code=status_code,
                response_chars=response_chars,
                chunk_intervals_s=chunk_intervals,
            )
        except (httpx.HTTPError, OSError) as exc:
            return RequestResult(
                request_id=request_id,
                success=False,
                latency_s=time.perf_counter() - started,
                status_code=status_code,
                error=str(exc)[:500],
            )
