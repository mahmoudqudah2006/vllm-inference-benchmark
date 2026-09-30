from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from typing import Any


def _gpu_info() -> list[dict[str, str]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []

    command = [
        executable,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return []

    gpus: list[dict[str, str]] = []
    for line in process.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            name, memory_mib, driver = parts
            gpus.append({"name": name, "memory_mib": memory_mib, "driver": driver})
    return gpus


def collect_system_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or None,
        "gpus": _gpu_info(),
    }
