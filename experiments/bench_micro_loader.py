"""Measure warmed per-pair latency for all DIV2K micro-loader degradations."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.micro_loader import Degradation, create_micro_dataloader

DEGRADATIONS: tuple[Degradation, ...] = (
    "noise",
    "super_resolution",
    "rain",
)


def measure_degradation(
    root: Path,
    degradation: Degradation,
    image_count: int,
    warmup_passes: int,
    repeats: int,
    max_ms: float,
) -> dict[str, object]:
    start = time.perf_counter()
    loader = create_micro_dataloader(
        root,
        degradation,
        image_count=image_count,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    initialization_ms = (time.perf_counter() - start) * 1000

    for _ in range(warmup_passes):
        for _pair in loader:
            pass

    latencies_ms = []
    for _ in range(repeats):
        iterator = iter(loader)
        for _ in range(len(loader)):
            start = time.perf_counter()
            next(iterator)
            latencies_ms.append((time.perf_counter() - start) * 1000)

    mean_ms = statistics.mean(latencies_ms)
    return {
        "degradation": degradation,
        "image_count": image_count,
        "warmup_passes": warmup_passes,
        "repeats": repeats,
        "initialization_ms": initialization_ms,
        "samples": len(latencies_ms),
        "mean_ms": mean_ms,
        "median_ms": statistics.median(latencies_ms),
        "max_ms": max(latencies_ms),
        "max_allowed_ms": max_ms,
        "pass": mean_ms < max_ms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="DIV2K_train_HR directory or its parent directory",
    )
    parser.add_argument("--image-count", type=int, default=10)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-ms", type=float, default=100.0)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optional path for machine-readable benchmark results",
    )
    args = parser.parse_args(argv)
    if args.image_count <= 0:
        parser.error("--image-count must be positive")
    if args.warmup_passes < 0:
        parser.error("--warmup-passes must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.max_ms <= 0:
        parser.error("--max-ms must be positive")

    results = [
        measure_degradation(
            args.data_root,
            degradation,
            args.image_count,
            args.warmup_passes,
            args.repeats,
            args.max_ms,
        )
        for degradation in DEGRADATIONS
    ]
    report = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "data_root": str(args.data_root.resolve()),
        "results": results,
        "pass": all(result["pass"] for result in results),
    }
    for result in results:
        print(
            f"{result['degradation']}: mean={result['mean_ms']:.2f} ms, "
            f"median={result['median_ms']:.2f} ms, "
            f"max={result['max_ms']:.2f} ms "
            f"(target < {args.max_ms:.2f} ms) "
            f"pass={result['pass']}"
        )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"results={args.json_output}")
    print(f"overall pass={report['pass']}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
