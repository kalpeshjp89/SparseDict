"""Compare PyTorch and Triton selected-value aggregation on CUDA."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernels.fused_topk_attn import tiled_topk_aggregate


def _torch_aggregate(
    top_scores: torch.Tensor,
    top_indices: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    weights = torch.softmax(top_scores.float(), dim=1)
    selected_values = values[top_indices]
    return torch.sum(weights.unsqueeze(-1) * selected_values, dim=1)


def _measure_ms(
    operation,
    *,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    durations = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            operation()
        end.record()
        end.synchronize()
        durations.append(start.elapsed_time(end) / repeats)
    return statistics.median(durations), max(durations)


def benchmark(
    *,
    query_counts: list[int],
    key_count: int,
    k: int,
    value_dim: int,
    warmup: int,
    repeats: int,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("The kernel benchmark requires a CUDA-enabled PyTorch runtime")
    if key_count <= 0 or value_dim <= 0 or not 1 <= k <= key_count:
        raise ValueError(
            "key_count/value_dim must be positive and k must be in [1, key_count]"
        )
    if any(count <= 0 for count in query_counts):
        raise ValueError("query counts must be positive")

    device = torch.device("cuda")
    results = []
    for query_count in query_counts:
        generator = torch.Generator(device=device).manual_seed(
            query_count * 10000 + key_count * 100 + k + value_dim
        )
        top_scores = torch.randn(
            (query_count, k),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        top_indices = torch.randint(
            key_count, (query_count, k), device=device, generator=generator
        )
        values = torch.randn(
            (key_count, value_dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )

        torch_output = _torch_aggregate(top_scores, top_indices, values)
        triton_output = tiled_topk_aggregate(top_scores, top_indices, values)
        torch.cuda.synchronize()
        max_abs_error = (torch_output - triton_output).abs().max().item()
        if max_abs_error > 1e-5:
            raise AssertionError(
                f"aggregation mismatch at N={query_count}: max error {max_abs_error}"
            )

        torch_median_ms, torch_max_ms = _measure_ms(
            lambda: _torch_aggregate(top_scores, top_indices, values),
            warmup=warmup,
            repeats=repeats,
        )
        triton_median_ms, triton_max_ms = _measure_ms(
            lambda: tiled_topk_aggregate(top_scores, top_indices, values),
            warmup=warmup,
            repeats=repeats,
        )
        results.append(
            {
                "query_count": query_count,
                "key_count": key_count,
                "k": k,
                "value_dim": value_dim,
                "max_abs_error": max_abs_error,
                "pytorch_median_ms": torch_median_ms,
                "pytorch_max_ms": torch_max_ms,
                "triton_median_ms": triton_median_ms,
                "triton_max_ms": triton_max_ms,
                "median_speedup": torch_median_ms / triton_median_ms,
            }
        )

    return {
        "device": torch.cuda.get_device_name(device),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "triton": __import__("triton").__version__,
        "operation": "selected-value softmax aggregation (top-k scores/indices precomputed)",
        "timing": "CUDA events; warmups excluded; median of three repeated batches",
        "warmup": warmup,
        "repeats_per_batch": repeats,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-counts", type=int, nargs="+", default=[1024, 4096, 16384]
    )
    parser.add_argument("--key-count", type=int, default=4096)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--json-output", type=str)
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    report = benchmark(
        query_counts=args.query_counts,
        key_count=args.key_count,
        k=args.k,
        value_dim=args.value_dim,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as output_file:
            output_file.write(serialized + "\n")


if __name__ == "__main__":
    main()
