"""Compare sparse-attention selection strategies on CUDA."""

from __future__ import annotations

import argparse
import math
import statistics

import torch

from kernels.fused_topk_attn import tiled_topk_aggregate
from kernels.hybrid_topk import hybrid_topk_attention
from models.modules.topk_select import sparse_topk_attention


def _reference(queries, keys, values, k):
    scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(queries.shape[-1])
    return sparse_topk_attention(scores, values, k).output


def _torch_topk_gather(queries, keys, values, k):
    scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(queries.shape[-1])
    top_scores, top_indices = torch.topk(scores, k, dim=-1, sorted=False)
    batch, heads, query_count, _ = top_scores.shape
    flattened_scores = top_scores.reshape(batch * heads * query_count, k)
    flattened_indices = top_indices.reshape(batch * heads * query_count, k)
    flattened_values = values.reshape(batch * heads, values.shape[2], values.shape[3])
    output = torch.empty(
        (batch * heads, query_count, values.shape[-1]),
        device=values.device,
        dtype=torch.float32,
    )
    for bh in range(batch * heads):
        start = bh * query_count
        stop = start + query_count
        output[bh] = tiled_topk_aggregate(
            flattened_scores[start:stop],
            flattened_indices[start:stop],
            flattened_values[bh],
        )
    return output.reshape(batch, heads, query_count, values.shape[-1])


def _chunked_torch_topk(queries, keys, values, k, chunk_size):
    batch, heads, query_count, feature_dim = queries.shape
    key_count = keys.shape[2]
    query_rows = batch * heads * query_count
    queries2 = queries.reshape(batch * heads, query_count, feature_dim)
    keys2 = keys.reshape(batch * heads, key_count, feature_dim)
    values2 = values.reshape(batch * heads, key_count, values.shape[-1])
    candidates = []
    candidate_indices = []
    for start in range(0, key_count, chunk_size):
        stop = min(start + chunk_size, key_count)
        scores = torch.matmul(queries2, keys2[:, start:stop].transpose(-2, -1))
        scores = scores / math.sqrt(feature_dim)
        local_k = min(k, stop - start)
        local_scores, local_indices = torch.topk(
            scores, local_k, dim=-1, sorted=False
        )
        candidates.append(local_scores.reshape(query_rows, local_k))
        candidate_indices.append(
            (local_indices + start).reshape(query_rows, local_k)
        )
    candidate_scores = torch.cat(candidates, dim=-1)
    candidate_index = torch.cat(candidate_indices, dim=-1)
    scores, order = torch.topk(candidate_scores, k, dim=-1, sorted=False)
    indices = candidate_index.gather(-1, order)
    output = torch.empty(
        (batch * heads, query_count, values.shape[-1]),
        device=values.device,
        dtype=torch.float32,
    )
    for bh in range(batch * heads):
        start = bh * query_count
        stop = start + query_count
        output[bh] = tiled_topk_aggregate(
            scores[start:stop], indices[start:stop], values2[bh]
        )
    return output.reshape(batch, heads, query_count, values.shape[-1])


def _measure(operation, warmup, repeats):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    timings = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            operation()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / repeats)
    return statistics.median(timings)


def _peak_mib(operation):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    operation()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / (1024 * 1024)


def benchmark(query_count, key_count, feature_dim, value_dim, k, warmup, repeats):
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA-enabled PyTorch runtime")
    generator = torch.Generator(device="cuda").manual_seed(548)
    queries = torch.randn(
        1, 1, query_count, feature_dim, device="cuda", generator=generator
    )
    keys = torch.randn(1, 1, key_count, feature_dim, device="cuda", generator=generator)
    values = torch.randn(1, 1, key_count, value_dim, device="cuda", generator=generator)
    reference = _reference(queries, keys, values, k)
    strategies = {
        "pytorch_reference": lambda: _reference(queries, keys, values, k),
        "full_gemm_torch_topk_triton_gather": lambda: _torch_topk_gather(
            queries, keys, values, k
        ),
        "option3_chunk1024_triton_merge": lambda: hybrid_topk_attention(
            queries, keys, values, k
        ),
    }
    for chunk_size in (256, 512, 1024, 2048, 4096):
        if chunk_size <= key_count:
            strategies[f"chunked_torch_topk_{chunk_size}"] = (
                lambda chunk_size=chunk_size: _chunked_torch_topk(
                    queries, keys, values, k, chunk_size
                )
            )

    result = []
    for name, operation in strategies.items():
        output = operation()
        torch.cuda.synchronize()
        error = (output - reference).abs().max().item()
        if error > 1e-5:
            raise AssertionError(f"{name} parity failure: max absolute error {error}")
        latency_ms = _measure(operation, warmup, repeats)
        peak_mib = _peak_mib(operation)
        result.append((name, latency_ms, peak_mib, error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--keys", type=int, default=4096)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--values", type=int, default=64)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print("GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__)
    print("strategy | ms | peak MiB | max abs err")
    for name, latency, memory, error in benchmark(
        args.queries,
        args.keys,
        args.features,
        args.values,
        args.k,
        args.warmup,
        args.repeats,
    ):
        print(f"{name} | {latency:.4f} | {memory:.1f} | {error:.3e}")


if __name__ == "__main__":
    main()
