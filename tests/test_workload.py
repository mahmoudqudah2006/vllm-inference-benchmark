import json
from pathlib import Path

import pytest

from vllm_inference_benchmark.workload import expand_prompts, load_prompts


def test_load_and_expand_prompts(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        "\n".join(json.dumps({"prompt": p}) for p in ["alpha", "beta"]),
        encoding="utf-8",
    )
    prompts = load_prompts(path)
    assert prompts == ["alpha", "beta"]
    assert expand_prompts(prompts, 5) == ["alpha", "beta", "alpha", "beta", "alpha"]


def test_load_prompts_rejects_missing_prompt(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text":"wrong field"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="prompt"):
        load_prompts(path)
