from __future__ import annotations

import json
from pathlib import Path


def load_prompts(path: Path) -> list[str]:
    """Load prompts from JSONL records containing a non-empty `prompt` field."""
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            prompt = record.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Line {line_number} must contain a non-empty string 'prompt'.")
            prompts.append(prompt.strip())

    if not prompts:
        raise ValueError(f"No prompts found in {path}.")
    return prompts


def expand_prompts(prompts: list[str], num_requests: int) -> list[str]:
    if num_requests <= 0:
        raise ValueError("num_requests must be greater than zero.")
    return [prompts[index % len(prompts)] for index in range(num_requests)]
