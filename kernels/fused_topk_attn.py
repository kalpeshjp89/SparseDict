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
SOFTMAX_TILE = 32


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

                value_is_nan = values != values
                partner_is_nan = partner_values != partner_values
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
    def _topk_value_aggregate(
        top_scores,
        top_indices,
        values,
        output,
        QUERY_COUNT: tl.constexpr,
        KEY_COUNT: tl.constexpr,
        VALUE_DIM: tl.constexpr,
        SCORE_ROW_STRIDE: tl.constexpr,
        SCORE_KEY_STRIDE: tl.constexpr,
        INDEX_ROW_STRIDE: tl.constexpr,
        INDEX_KEY_STRIDE: tl.constexpr,
        VALUE_ROW_STRIDE: tl.constexpr,
        VALUE_FEATURE_STRIDE: tl.constexpr,
        OUTPUT_ROW_STRIDE: tl.constexpr,
        OUTPUT_FEATURE_STRIDE: tl.constexpr,
        SOFTMAX_BLOCK: tl.constexpr,
        FEATURE_BLOCK: tl.constexpr,
    ):
        query = tl.program_id(0)
        features = tl.program_id(1) * FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)
        key_positions = tl.arange(0, SOFTMAX_BLOCK)
        running_max = tl.full((), -float("inf"), tl.float32)
        running_sum = tl.full((), 0.0, tl.float32)
        accumulator = tl.zeros((FEATURE_BLOCK,), tl.float32)

        for block_start in range(tl.cdiv(KEY_COUNT, SOFTMAX_BLOCK)):
            positions = block_start * SOFTMAX_BLOCK + key_positions
            logits = tl.load(
                top_scores
                + query * SCORE_ROW_STRIDE
                + positions * SCORE_KEY_STRIDE,
                mask=(query < QUERY_COUNT) & (positions < KEY_COUNT),
                other=-float("inf"),
            ).to(tl.float32)
            selected_indices = tl.load(
                top_indices
                + query * INDEX_ROW_STRIDE
                + positions * INDEX_KEY_STRIDE,
                mask=(query < QUERY_COUNT) & (positions < KEY_COUNT),
                other=0,
            ).to(tl.int32)

            block_max = tl.max(logits, axis=0)
            new_max = tl.maximum(running_max, block_max)
            old_scale = tl.exp(running_max - new_max)
            weights = tl.exp(logits - new_max)
            running_sum = running_sum * old_scale + tl.sum(weights, axis=0)

            selected_values = tl.load(
                values
                + selected_indices[:, None] * VALUE_ROW_STRIDE
                + features[None, :] * VALUE_FEATURE_STRIDE,
                mask=(query < QUERY_COUNT)
                & (positions[:, None] < KEY_COUNT)
                & (features[None, :] < VALUE_DIM),
                other=0.0,
            ).to(tl.float32)
            accumulator = (
                accumulator * old_scale
                + tl.sum(weights[:, None] * selected_values, axis=0)
            )
            running_max = new_max

        aggregated = accumulator / running_sum
        tl.store(
            output + query * OUTPUT_ROW_STRIDE + features * OUTPUT_FEATURE_STRIDE,
            aggregated,
            mask=(query < QUERY_COUNT) & (features < VALUE_DIM),
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


def tiled_topk_aggregate(
    top_scores: torch.Tensor,
    top_indices: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Softmax-weight gathered value rows for each query's selected top-k.

    The selected score vector is normalized stably in float32; only the chosen
    value rows are gathered, and the output is accumulated in float32.
    """
    if top_scores.ndim != 2 or top_indices.ndim != 2:
        raise ValueError("top_scores and top_indices must be rank-2 tensors")
    if values.ndim != 2:
        raise ValueError("values must be a rank-2 (keys, features) tensor")
    if top_scores.shape != top_indices.shape:
        raise ValueError("top_scores and top_indices must have identical shapes")
    if top_scores.size(0) == 0 or top_scores.size(1) == 0 or values.size(1) == 0:
        raise ValueError("query, top-k, and value feature dimensions must be non-empty")
    if values.size(0) == 0:
        raise ValueError("values must contain at least one key row")
    if top_scores.device.type != "cuda" or top_indices.device.type != "cuda":
        raise ValueError("tiled_topk_aggregate requires CUDA tensors")
    if values.device.type != "cuda":
        raise ValueError("tiled_topk_aggregate requires CUDA values")
    if top_scores.device != top_indices.device or top_scores.device != values.device:
        raise ValueError("scores, indices, and values must be on the same CUDA device")
    if top_scores.dtype not in (torch.float16, torch.float32):
        raise ValueError("top_scores must have float16 or float32 dtype")
    if values.dtype not in (torch.float16, torch.float32):
        raise ValueError("values must have float16 or float32 dtype")
    if top_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("top_indices must have int32 or int64 dtype")
    if top_scores.stride(1) != 1 or top_indices.stride(1) != 1:
        raise ValueError("top-k dimensions must be contiguous")
    if values.stride(1) != 1:
        raise ValueError("value feature dimension must be contiguous")
    if triton is None:
        raise RuntimeError(
            "Triton is required for tiled_topk_aggregate; install a "
            "Triton-enabled PyTorch environment"
        )

    query_count, key_count = top_scores.shape
    value_dim = values.size(1)
    output = torch.empty(
        (query_count, value_dim), device=values.device, dtype=torch.float32
    )
    feature_block = min(32, triton.next_power_of_2(value_dim))
    grid = (query_count, triton.cdiv(value_dim, feature_block))
    _topk_value_aggregate[grid](
        top_scores,
        top_indices,
        values,
        output,
        query_count,
        key_count,
        value_dim,
        top_scores.stride(0),
        top_scores.stride(1),
        top_indices.stride(0),
        top_indices.stride(1),
        values.stride(0),
        values.stride(1),
        output.stride(0),
        output.stride(1),
        SOFTMAX_TILE,
        feature_block,
        num_warps=4,
    )
    return output
