"""Tiled query-key affinity kernel used as the Phase 5 Triton foundation."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

QUERY_TILE = 64
KEY_TILE = 64
FEATURE_TILE = 32


if triton is not None:

    @triton.jit
    def _affinity_tiles(
        queries,
        keys,
        scores,
        QUERY_COUNT: tl.constexpr,
        KEY_COUNT: tl.constexpr,
        FEATURE_COUNT: tl.constexpr,
        QUERY_STRIDE: tl.constexpr,
        QUERY_FEATURE_STRIDE: tl.constexpr,
        KEY_STRIDE: tl.constexpr,
        KEY_FEATURE_STRIDE: tl.constexpr,
        QUERY_BLOCK: tl.constexpr,
        KEY_BLOCK: tl.constexpr,
        FEATURE_BLOCK: tl.constexpr,
    ):
        query_rows = tl.program_id(0) * QUERY_BLOCK + tl.arange(0, QUERY_BLOCK)
        key_rows = tl.program_id(1) * KEY_BLOCK + tl.arange(0, KEY_BLOCK)
        feature_cols = tl.arange(0, FEATURE_BLOCK)
        accum = tl.full((QUERY_BLOCK, KEY_BLOCK), 0.0, tl.float32)

        for feature_start in range(tl.cdiv(FEATURE_COUNT, FEATURE_BLOCK)):
            features = feature_start * FEATURE_BLOCK + feature_cols
            query_tile = tl.load(
                queries
                + query_rows[:, None] * QUERY_STRIDE
                + features[None, :] * QUERY_FEATURE_STRIDE,
                mask=(query_rows[:, None] < QUERY_COUNT)
                & (features[None, :] < FEATURE_COUNT),
                other=0.0,
            )
            key_tile = tl.load(
                keys
                + key_rows[:, None] * KEY_STRIDE
                + features[None, :] * KEY_FEATURE_STRIDE,
                mask=(key_rows[:, None] < KEY_COUNT)
                & (features[None, :] < FEATURE_COUNT),
                other=0.0,
            )
            accum += tl.dot(
                query_tile.to(tl.float32),
                tl.trans(key_tile.to(tl.float32)),
                input_precision="ieee",
            )

        tl.store(
            scores + query_rows[:, None] * KEY_COUNT + key_rows[None, :],
            accum,
            mask=(query_rows[:, None] < QUERY_COUNT)
            & (key_rows[None, :] < KEY_COUNT),
        )


def tiled_affinity(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Compute ``queries @ keys.T`` with 64x64 query/key program tiles.

    Queries and keys have shape ``(N, D)`` and ``(M, D)`` respectively.
    Input feature dimensions are the contiguous inner dimension, enabling
    coalesced loads for both matrices. The returned dense scores are an
    intermediate foundation; top-k selection and value aggregation are added
    in later Phase 5 tasks.
    """
    if queries.ndim != 2 or keys.ndim != 2:
        raise ValueError("queries and keys must be rank-2 (items, features) tensors")
    if queries.size(1) != keys.size(1):
        raise ValueError(
            f"query/key feature sizes must match; got "
            f"{queries.size(1)} and {keys.size(1)}"
        )
    if queries.size(0) == 0 or keys.size(0) == 0 or queries.size(1) == 0:
        raise ValueError("query, key, and feature dimensions must be non-empty")
    if queries.stride(1) != 1 or keys.stride(1) != 1:
        raise ValueError("query and key feature dimensions must be contiguous")
    if queries.device.type != "cuda" or keys.device.type != "cuda":
        raise ValueError("tiled_affinity requires CUDA query and key tensors")
    if queries.device != keys.device:
        raise ValueError("queries and keys must be on the same CUDA device")
    if queries.dtype != keys.dtype or queries.dtype not in (
        torch.float16,
        torch.float32,
    ):
        raise ValueError("queries and keys must share a float16 or float32 dtype")
    if triton is None:
        raise RuntimeError(
            "Triton is required for tiled_affinity; install a Triton-enabled "
            "PyTorch environment"
        )

    query_count, feature_count = queries.shape
    key_count = keys.size(0)
    scores = torch.empty(
        (query_count, key_count), device=queries.device, dtype=torch.float32
    )
    grid = (
        triton.cdiv(query_count, QUERY_TILE),
        triton.cdiv(key_count, KEY_TILE),
    )
    _affinity_tiles[grid](
        queries,
        keys,
        scores,
        query_count,
        key_count,
        feature_count,
        queries.stride(0),
        queries.stride(1),
        keys.stride(0),
        keys.stride(1),
        QUERY_TILE,
        KEY_TILE,
        FEATURE_TILE,
        num_warps=4,
    )
    return scores
