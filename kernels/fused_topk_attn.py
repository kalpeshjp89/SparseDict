"""Tiled affinity and in-register top-k kernels for Phase 5."""

from __future__ import annotations

import math

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
    def _bitonic_topk(
        scores,
        top_scores,
        top_indices,
        ROW_COUNT: tl.constexpr,
        KEY_COUNT: tl.constexpr,
        K: tl.constexpr,
        SCORE_ROW_STRIDE: tl.constexpr,
        SCORE_KEY_STRIDE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        SORT_STAGES: tl.constexpr,
    ):
        row = tl.program_id(0)
        positions = tl.arange(0, BLOCK_SIZE)
        values = tl.load(
            scores + row * SCORE_ROW_STRIDE + positions * SCORE_KEY_STRIDE,
            mask=(row < ROW_COUNT) & (positions < KEY_COUNT),
            other=-float("inf"),
        ).to(tl.float32)
        indices = positions

        for stage in range(1, SORT_STAGES + 1):
            stage_mask = 1 << stage
            for stride_log in range(stage - 1, -1, -1):
                stride = 1 << stride_log
                partner_positions = positions ^ stride
                partner_values = tl.gather(values, partner_positions, axis=0)
                partner_indices = tl.gather(indices, partner_positions, axis=0)

                value_is_nan = tl.isnan(values)
                partner_is_nan = tl.isnan(partner_values)
                both_nan = value_is_nan & partner_is_nan
                regular_better = (
                    ~value_is_nan
                    & ~partner_is_nan
                    & (
                        (values > partner_values)
                        | ((values == partner_values) & (indices < partner_indices))
                    )
                )
                self_is_better = (
                    (value_is_nan & ~partner_is_nan)
                    | (both_nan & (indices < partner_indices))
                    | regular_better
                )

                lower_in_pair = (positions & stride) == 0
                ascending_group = (positions & stage_mask) != 0
                take_better = lower_in_pair != ascending_group
                choose_self = self_is_better == take_better
                values = tl.where(choose_self, values, partner_values)
                indices = tl.where(choose_self, indices, partner_indices)

        tl.store(
            top_scores + row * K + positions,
            values,
            mask=(row < ROW_COUNT) & (positions < K),
        )
        tl.store(
            top_indices + row * K + positions,
            indices.to(tl.int64),
            mask=(row < ROW_COUNT) & (positions < K),
        )

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


def tiled_topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select each row's largest ``k`` values with an in-register bitonic sort.

    The selector reads the score matrix and writes only the selected scores and
    indices. Equal scores are ordered by ascending column index; NaNs sort ahead
    of numeric values. As with ``torch.topk``, index order for tied values is
    implementation-dependent, so exact cross-backend index comparisons should
    use inputs with distinct values.
    """
    if scores.ndim != 2:
        raise ValueError("scores must be a rank-2 (queries, keys) tensor")
    if scores.size(0) == 0 or scores.size(1) == 0:
        raise ValueError("score dimensions must be non-empty")
    if (
        not isinstance(k, int)
        or isinstance(k, bool)
        or not 1 <= k <= scores.size(1)
    ):
        raise ValueError(f"k must be an integer in [1, {scores.size(1)}]")
    if scores.device.type != "cuda":
        raise ValueError("tiled_topk requires CUDA scores")
    if scores.dtype not in (torch.float16, torch.float32):
        raise ValueError("scores must have float16 or float32 dtype")
    if triton is None:
        raise RuntimeError(
            "Triton is required for tiled_topk; install a Triton-enabled "
            "PyTorch environment"
        )

    row_count, key_count = scores.shape
    block_size = triton.next_power_of_2(key_count)
    top_scores = torch.empty((row_count, k), device=scores.device, dtype=scores.dtype)
    top_indices = torch.empty((row_count, k), device=scores.device, dtype=torch.int64)
    _bitonic_topk[(row_count,)](
        scores,
        top_scores,
        top_indices,
        row_count,
        key_count,
        k,
        scores.stride(0),
        scores.stride(1),
        block_size,
        int(math.log2(block_size)),
        num_warps=4,
    )
    return top_scores, top_indices
