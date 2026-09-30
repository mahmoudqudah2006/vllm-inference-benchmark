from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_column(path: Path, column: str) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get(column)
            if value:
                values.append(float(value))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark latency distributions.")
    parser.add_argument("csv", type=Path, help="Path to requests.csv")
    parser.add_argument("--output", type=Path, default=Path("latency_distribution.png"))
    args = parser.parse_args()

    latency_ms = [value * 1000 for value in read_column(args.csv, "latency_s")]
    if not latency_ms:
        raise SystemExit("No latency values found.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(latency_ms, bins=min(30, max(5, len(latency_ms) // 2)))
    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("Requests")
    ax.set_title("LLM Inference Latency Distribution")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
