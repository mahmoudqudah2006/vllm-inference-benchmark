import httpx
import pytest

from vllm_inference_benchmark.client import ClientSettings, OpenAICompatibleClient


@pytest.mark.asyncio
async def test_streaming_chat_request_collects_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = "\n\n".join(
            [
                'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                'data: {"choices":[{"delta":{"content":" world"}}]}',
                (
                    'data: {"choices":[],"usage":{'
                    '"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}'
                ),
                "data: [DONE]",
            ]
        )
        return httpx.Response(200, text=body)

    client = OpenAICompatibleClient(ClientSettings("http://test"))
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.run_chat_request(
            request_id=7,
            prompt="Say hello",
            model="demo-model",
            max_tokens=16,
            temperature=0.0,
        )
    finally:
        await client._client.aclose()

    assert result.success is True
    assert result.status_code == 200
    assert result.response_chars == len("Hello world")
    assert result.prompt_tokens == 4
    assert result.completion_tokens == 2
    assert result.total_tokens == 6
    assert result.ttft_s is not None
    assert result.tpot_s is not None


@pytest.mark.asyncio
async def test_model_resolution() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "Qwen/demo"}]})

    client = OpenAICompatibleClient(ClientSettings("http://test"))
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert await client.resolve_model() == "Qwen/demo"
    finally:
        await client._client.aclose()
