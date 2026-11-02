"""Compare the fused Triton top-k attention path against PyTorch end to end."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernels.fused_topk_attn import fused_topk_attention
from models.modules.topk_select import sparse_topk_attention


def _torch_reference(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Materialize dense scores, run top-k, softmax, and gather in PyTorch."""
    scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(queries.shape[-1])
    return sparse_topk_attention(scores, values, k).output


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


def _allocated_delta_mib(operation) -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    operation()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / (1024 * 1024)


def benchmark(
    *,
    query_counts: list[int],
    key_count: int,
    k: int,
    query_dim: int,
    value_dim: int,
    batch: int,
    heads: int,
    warmup: int,
    repeats: int,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("The end-to-end benchmark requires a CUDA-enabled PyTorch runtime")
    if key_count <= 0 or value_dim <= 0 or query_dim <= 0:
        raise ValueError("key_count/value_dim/query_dim must be positive")
    if not 1 <= k <= key_count:
        raise ValueError(f"k must be in [1, {key_count}]; got {k}")
    if any(count <= 0 for count in query_counts):
        raise ValueError("query counts must be positive")
    if batch <= 0 or heads <= 0:
        raise ValueError("batch and heads must be positive")

    device = torch.device("cuda")
    results = []
    for query_count in query_counts:
        generator = torch.Generator(device=device).manual_seed(
            query_count * 10000 + key_count * 100 + k + value_dim
        )
        queries = torch.randn(
            (batch, heads, query_count, query_dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        keys = torch.randn(
            (batch, heads, key_count, query_dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        values = torch.randn(
            (batch, heads, key_count, value_dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )

        reference_output = _torch_reference(queries, keys, values, k)
        fused_output = fused_topk_attention(queries, keys, values, k)
        torch.cuda.synchronize()
        max_abs_error = (reference_output - fused_output).abs().max().item()
        if max_abs_error > 1e-5:
            raise AssertionError(
                f"end-to-end mismatch at N={query_count}: max error {max_abs_error}"
            )

        reference_median_ms, reference_max_ms = _measure_ms(
            lambda: _torch_reference(queries, keys, values, k),
            warmup=warmup,
            repeats=repeats,
        )
        fused_median_ms, fused_max_ms = _measure_ms(
            lambda: fused_topk_attention(queries, keys, values, k),
            warmup=warmup,
            repeats=repeats,
        )
        reference_mib = _allocated_delta_mib(
            lambda: _torch_reference(queries, keys, values, k)
        )
        fused_mib = _allocated_delta_mib(
            lambda: fused_topk_attention(queries, keys, values, k)
        )
        results.append(
            {
                "query_count": query_count,
                "key_count": key_count,
                "k": k,
                "query_dim": query_dim,
                "value_dim": value_dim,
                "batch": batch,
                "heads": heads,
                "max_abs_error": max_abs_error,
                "reference_median_ms": reference_median_ms,
                "reference_max_ms": reference_max_ms,
                "fused_median_ms": fused_median_ms,
                "fused_max_ms": fused_max_ms,
                "median_speedup": reference_median_ms / fused_median_ms,
                "reference_peak_delta_mib": round(reference_mib, 3),
                "fused_peak_delta_mib": round(fused_mib, 3),
            }
        )

    return {
        "device": torch.cuda.get_device_name(device),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "triton": __import__("triton").__version__,
        "operation": (
            "end-to-end sparse top-k attention: chunked cuBLAS affinity, "
            "chunk-local CUDA top-k, and fused Triton global top-k softmax-gather; "
            "PyTorch reference materializes the dense N x M score matrix"
        ),
        "timing": "CUDA events; warmups excluded; median of three repeated batches",
        "memory": "torch.cuda.max_memory_allocated delta over one forward",
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
    parser.add_argument("--query-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    payload = benchmark(
        query_counts=args.query_counts,
        key_count=args.key_count,
        k=args.k,
        query_dim=args.query_dim,
        value_dim=args.value_dim,
        batch=args.batch,
        heads=args.heads,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    header = (
        f"{'N':>7} | {'PyTorch (ms)':>13} | {'Fused (ms)':>11} | "
        f"{'Speedup':>8} | {'Ref MiB':>9} | {'Fused MiB':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in payload["results"]:
        print(
            f"{row['query_count']:>7} | {row['reference_median_ms']:>13.4f} | "
            f"{row['fused_median_ms']:>11.4f} | {row['median_speedup']:>7.2f}x | "
            f"{row['reference_peak_delta_mib']:>9.1f} | {row['fused_peak_delta_mib']:>9.1f}"
        )

    args.json_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.json_dir / "fused_e2e_benchmark.json"
    stamp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {stamp}")


if __name__ == "__main__":
    main()
