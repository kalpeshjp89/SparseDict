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

# Task 5.4b tiles for the fused program-per-query-block kernels.
FUSED_QUERY_BLOCK = 32
FUSED_BACKWARD_QUERY_BLOCK = 16
FUSED_KEY_BLOCK = 64
FUSED_FEATURE_BLOCK = 32
FUSED_KCHUNK = 16

_FUSED_DTYPE_MESSAGE = "fused path supports float32 and float64 inputs"

_FUSED_DTYPE_MAP = (
    {torch.float32: tl.float32, torch.float64: tl.float64} if triton is not None else {}
)


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


# --- Task 5.4b: unified fused top-k attention with analytical backward ------

if triton is not None:

    @triton.jit
    def _fused_affinity_topk(
        queries,
        keys,
        top_scores,
        top_indices,
        QUERY_COUNT: tl.constexpr,
        KEY_COUNT: tl.constexpr,
        FEATURE_COUNT: tl.constexpr,
        QUERY_BH_STRIDE: tl.constexpr,
        QUERY_ROW_STRIDE: tl.constexpr,
        KEY_BH_STRIDE: tl.constexpr,
        KEY_ROW_STRIDE: tl.constexpr,
        SCORE_ROW_STRIDE: tl.constexpr,
        INDEX_ROW_STRIDE: tl.constexpr,
        K: tl.constexpr,
        ARR_LEN: tl.constexpr,
        SORT_STAGES: tl.constexpr,
        QUERY_BLOCK: tl.constexpr,
        FEATURE_BLOCK: tl.constexpr,
        KEY_BLOCK: tl.constexpr,
        USE_DOT: tl.constexpr,
        PAD_INDEX: tl.constexpr,
        DTYPE: tl.constexpr,
    ):
        """Fused tiled affinity + running in-register top-k (design F1).

        Each program owns a query block on one ``(batch, head)`` slice and
        streams all keys. The candidate register array is ``ARR_LEN`` lanes
        wide: lanes ``[0, K)`` carry the running top-k gathered from the
        buffer registers, lanes ``[K, K + KEY_BLOCK)`` hold the fresh tile
        affinities loaded directly at their lane, and the remainder is padded
        with ``-inf`` / ``PAD_INDEX``. A full bitonic sort (same comparator
        and ascending-index tie-break as ``_bitonic_topk``) keeps the buffer
        the descending top-``ARR_LEN`` of all keys seen so far, so lanes
        ``[0, K)`` hold the global top-k after the last tile.

        Fresh affinity lanes map to global key ``tile * KEY_BLOCK + lane -
        K``; the first ``K`` lanes are masked loads that never alias real
        keys, and invalid lanes are set to ``-inf`` so padded candidates
        contribute exactly zero softmax mass downstream.
        """
        rows = tl.program_id(0) * QUERY_BLOCK + tl.arange(0, QUERY_BLOCK)
        row_mask = rows < QUERY_COUNT
        bh = tl.program_id(1)
        lanes = tl.broadcast_to(tl.arange(0, ARR_LEN)[None, :], (QUERY_BLOCK, ARR_LEN))
        lanes_1d = tl.arange(0, ARR_LEN)
        buf_src_lanes = tl.minimum(lanes, K - 1)
        buf_val = tl.full((QUERY_BLOCK, ARR_LEN), -float("inf"), DTYPE)
        buf_idx = tl.full((QUERY_BLOCK, ARR_LEN), PAD_INDEX, tl.int32)

        for tile in range(0, tl.cdiv(KEY_COUNT, KEY_BLOCK)):
            key_rows = tile * KEY_BLOCK - K + lanes_1d
            key_valid = (key_rows >= 0) & (key_rows < KEY_COUNT)
            accum = tl.full((QUERY_BLOCK, ARR_LEN), 0.0, DTYPE)
            for chunk in range(0, tl.cdiv(FEATURE_COUNT, FEATURE_BLOCK)):
                cols = chunk * FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)
                cols_valid = cols < FEATURE_COUNT
                q_tile = tl.load(
                    queries
                    + bh * QUERY_BH_STRIDE
                    + rows[:, None] * QUERY_ROW_STRIDE
                    + cols[None, :],
                    mask=row_mask[:, None] & cols_valid[None, :],
                    other=0.0,
                )
                k_tile = tl.load(
                    keys
                    + bh * KEY_BH_STRIDE
                    + key_rows[:, None] * KEY_ROW_STRIDE
                    + cols[None, :],
                    mask=key_valid[:, None] & cols_valid[None, :],
                    other=0.0,
                )
                if USE_DOT:
                    accum += tl.dot(
                        q_tile.to(DTYPE),
                        tl.trans(k_tile.to(DTYPE)),
                        input_precision="ieee",
                    )
                else:
                    accum += tl.sum(
                        q_tile[:, None, :] * k_tile[None, :, :], axis=2
                    )

            fresh_val = tl.where(key_valid[None, :], accum, -float("inf"))
            fresh_idx = tl.where(key_valid[None, :], key_rows, PAD_INDEX).to(tl.int32)
            cand_val = tl.where(
                lanes < K, tl.gather(buf_val, buf_src_lanes, axis=1), fresh_val
            )
            cand_idx = tl.where(
                lanes < K, tl.gather(buf_idx, buf_src_lanes, axis=1), fresh_idx
            )

            positions = lanes
            values = cand_val
            indices = cand_idx
            for stage in range(1, SORT_STAGES + 1):
                stage_mask = 1 << stage
                for stride_log in range(stage - 1, -1, -1):
                    stride = 1 << stride_log
                    partner_positions = positions ^ stride
                    partner_values = tl.gather(values, partner_positions, axis=1)
                    partner_indices = tl.gather(indices, partner_positions, axis=1)

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
            buf_val = values
            buf_idx = indices

        out_rows = bh * QUERY_COUNT + rows
        tl.store(
            top_scores + out_rows[:, None] * SCORE_ROW_STRIDE + lanes,
            buf_val,
            mask=row_mask[:, None] & (lanes < K),
        )
        tl.store(
            top_indices + out_rows[:, None] * INDEX_ROW_STRIDE + lanes,
            buf_idx,
            mask=row_mask[:, None] & (lanes < K),
        )

    @triton.jit
    def _backward_dq(
        queries,
        keys,
        values,
        grad_out,
        saved_out,
        top_scores,
        top_indices,
        lse,
        d_scores,
        dq,
        QUERY_COUNT: tl.constexpr,
        KEY_COUNT: tl.constexpr,
        FEAT_Q: tl.constexpr,
        FEAT_V: tl.constexpr,
        QUERY_BH_STRIDE: tl.constexpr,
        QUERY_ROW_STRIDE: tl.constexpr,
        KEY_BH_STRIDE: tl.constexpr,
        KEY_ROW_STRIDE: tl.constexpr,
        VALUE_BH_STRIDE: tl.constexpr,
        VALUE_ROW_STRIDE: tl.constexpr,
        GO_BH_STRIDE: tl.constexpr,
        GO_ROW_STRIDE: tl.constexpr,
        OUT_BH_STRIDE: tl.constexpr,
        OUT_ROW_STRIDE: tl.constexpr,
        TS_ROW_STRIDE: tl.constexpr,
        TI_ROW_STRIDE: tl.constexpr,
        DS_ROW_STRIDE: tl.constexpr,
        DQ_ROW_STRIDE: tl.constexpr,
        K: tl.constexpr,
        KP: tl.constexpr,
        KCHUNK: tl.constexpr,
        QUERY_BLOCK: tl.constexpr,
        FEATURE_BLOCK: tl.constexpr,
        DTYPE: tl.constexpr,
    ):
        """Backward B1: exact, atomic-free ``dQ`` plus the ``dS (N, K)`` scratch.

        Recomputes ``P = exp(S - L)`` on the saved sparse support, builds
        ``G = dO . V`` per selected pair, and writes
        ``dS = P * (G - delta)`` with ``delta_i = dot(dO_i, O_i)``. The
        ``dS`` scratch feeds kernel B2 (which loads it in a separate
        launch, so no same-program store/load hazard exists). ``dQ``
        recomputes ``dS`` inline per k-chunk (never re-loading the scratch
        within this program) and gathers ``K[I_s]`` rows over feature
        chunks — no dense reads or writes anywhere.
        """
        rows = tl.program_id(0) * QUERY_BLOCK + tl.arange(0, QUERY_BLOCK)
        row_mask = rows < QUERY_COUNT
        bh = tl.program_id(1)
        out_rows = bh * QUERY_COUNT + rows

        delta = tl.full((QUERY_BLOCK,), 0.0, DTYPE)
        for chunk in range(0, tl.cdiv(FEAT_V, FEATURE_BLOCK)):
            vcols = chunk * FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)
            vcols_valid = vcols < FEAT_V
            do_tile = tl.load(
                grad_out
                + bh * GO_BH_STRIDE
                + rows[:, None] * GO_ROW_STRIDE
                + vcols[None, :],
                mask=row_mask[:, None] & vcols_valid[None, :],
                other=0.0,
            ).to(DTYPE)
            out_tile = tl.load(
                saved_out
                + bh * OUT_BH_STRIDE
                + rows[:, None] * OUT_ROW_STRIDE
                + vcols[None, :],
                mask=row_mask[:, None] & vcols_valid[None, :],
                other=0.0,
            ).to(DTYPE)
            delta += tl.sum(do_tile * out_tile, axis=1)

        for kc in range(0, tl.cdiv(K, KCHUNK)):
            kcols = kc * KCHUNK + tl.arange(0, KCHUNK)
            kmask = kcols < K
            scores = tl.load(
                top_scores + out_rows[:, None] * TS_ROW_STRIDE + kcols[None, :],
                mask=row_mask[:, None] & kmask[None, :],
                other=-float("inf"),
            ).to(DTYPE)
            indices = tl.load(
                top_indices + out_rows[:, None] * TI_ROW_STRIDE + kcols[None, :],
                mask=row_mask[:, None] & kmask[None, :],
                other=0,
            ).to(tl.int32)
            lse_vals = tl.load(lse + out_rows, mask=row_mask, other=0.0).to(DTYPE)
            probs = tl.exp(scores - lse_vals[:, None])
            g_vals = tl.full((QUERY_BLOCK, KCHUNK), 0.0, DTYPE)
            for vchunk in range(0, tl.cdiv(FEAT_V, FEATURE_BLOCK)):
                vcols = vchunk * FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)
                vcols_valid = vcols < FEAT_V
                sel_values = tl.load(
                    values
                    + bh * VALUE_BH_STRIDE
                    + indices[:, :, None] * VALUE_ROW_STRIDE
                    + vcols[None, None, :],
                    mask=row_mask[:, None, None]
                    & kmask[:, None]
                    & vcols_valid[None, None, :],
                    other=0.0,
                ).to(DTYPE)
                do_tile = tl.load(
                    grad_out
                    + bh * GO_BH_STRIDE
                    + rows[:, None] * GO_ROW_STRIDE
                    + vcols[None, :],
                    mask=row_mask[:, None] & vcols_valid[None, :],
                    other=0.0,
                ).to(DTYPE)
                g_vals += tl.sum(sel_values * do_tile[:, None, :], axis=2)
            d_s = probs * (g_vals - delta[:, None])
            tl.store(
                d_scores + out_rows[:, None] * DS_ROW_STRIDE + kcols[None, :],
                d_s,
                mask=row_mask[:, None] & kmask[None, :],
            )

        for fchunk in range(0, tl.cdiv(FEAT_Q, FEATURE_BLOCK)):
            fcols = fchunk * FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)
            fmask = fcols < FEAT_Q
            dq_acc = tl.full((QUERY_BLOCK, FEATURE_BLOCK), 0.0, DTYPE)
            for kc2 in range(0, tl.cdiv(KP, KCHUNK)):
                kk = kc2 * KCHUNK + tl.arange(0, KCHUNK)
                kk_mask = kk < K
                scores = tl.load(
                    top_scores + out_rows[:, None] * TS_ROW_STRIDE + kk[None, :],
                    mask=row_mask[:, None] & kk_mask[None, :],
                    other=-float("inf"),
                ).to(DTYPE)
                indices = tl.load(
                    top_indices + out_rows[:, None] * TI_ROW_STRIDE + kk[None, :],
                    mask=row_mask[:, None] & kk_mask[None, :],
                    other=0,
                ).to(tl.int32)
                lse_vals = tl.load(lse + out_rows, mask=row_mask, other=0.0).to(DTYPE)
                probs = tl.exp(scores - lse_vals[:, None])
                g_vals = tl.full((QUERY_BLOCK, KCHUNK), 0.0, DTYPE)
                for vchunk in range(0, tl.cdiv(FEAT_V, FEATURE_BLOCK)):
                    vcols = vchunk * FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)
                    vcols_valid = vcols < FEAT_V
                    sel_values = tl.load(
                        values
                        + bh * VALUE_BH_STRIDE
                        + indices[:, :, None] * VALUE_ROW_STRIDE
                        + vcols[None, None, :],
                        mask=row_mask[:, None, None]
                        & kk_mask[:, None]
                        & vcols_valid[None, None, :],
                        other=0.0,
                    ).to(DTYPE)
                    do_tile = tl.load(
                        grad_out
                        + bh * GO_BH_STRIDE
                        + rows[:, None] * GO_ROW_STRIDE
                        + vcols[None, :],
                        mask=row_mask[:, None] & vcols_valid[None, :],
                        other=0.0,
                    ).to(DTYPE)
                    g_vals += tl.sum(sel_values * do_tile[:, None, :], axis=2)
                d_s = probs * (g_vals - delta[:, None])
                sel_keys = tl.load(
                    keys
                    + bh * KEY_BH_STRIDE
                    + indices[:, :, None] * KEY_ROW_STRIDE
                    + fcols[None, None, :],
                    mask=row_mask[:, None, None]
                    & kk_mask[:, None]
                    & fmask[None, None, :],
                    other=0.0,
                ).to(DTYPE)
                dq_acc += tl.sum(d_s[:, :, None] * sel_keys, axis=1)
            tl.store(
                dq
                + bh * QUERY_BH_STRIDE
                + rows[:, None] * DQ_ROW_STRIDE
                + fcols[None, :],
                dq_acc,
                mask=row_mask[:, None] & fmask[None, :],
            )

    @triton.jit
    def _backward_dkv(
        queries,
        grad_out,
        top_scores,
        top_indices,
        lse,
        d_scores,
        d_keys,
        d_values,
        QUERY_COUNT: tl.constexpr,
        KEY_COUNT: tl.constexpr,
        FEAT_Q: tl.constexpr,
        FEAT_V: tl.constexpr,
        KEY_BH_STRIDE: tl.constexpr,
        KEY_ROW_STRIDE: tl.constexpr,
        VALUE_BH_STRIDE: tl.constexpr,
        VALUE_ROW_STRIDE: tl.constexpr,
        GO_BH_STRIDE: tl.constexpr,
        GO_ROW_STRIDE: tl.constexpr,
        TS_ROW_STRIDE: tl.constexpr,
        TI_ROW_STRIDE: tl.constexpr,
        DS_ROW_STRIDE: tl.constexpr,
        K: tl.constexpr,
        KP: tl.constexpr,
        KCHUNK: tl.constexpr,
        QUERY_BLOCK: tl.constexpr,
        FEATURE_BLOCK: tl.constexpr,
        DTYPE: tl.constexpr,
    ):
        """Backward B2: block-sparse atomic scatter of ``dK`` and ``dV``.

        Consumes only saved top-k pairs: recomputes ``P = exp(S - L)``,
        loads the precomputed ``dS`` scratch written by kernel B1 (a
        separate launch), and atomically adds ``dV[I] += P * dO`` and
        ``dK[I] += dS * Q``. Atomic ordering across query rows is the only
        nondeterminism, at ulp scale, orders below the 1e-4 gate.
        """
        rows = tl.program_id(0) * QUERY_BLOCK + tl.arange(0, QUERY_BLOCK)
        row_mask = rows < QUERY_COUNT
        bh = tl.program_id(1)
        out_rows = bh * QUERY_COUNT + rows

        for kc in range(0, tl.cdiv(KP, KCHUNK)):
            kcols = kc * KCHUNK + tl.arange(0, KCHUNK)
            kmask = kcols < K
            scores = tl.load(
                top_scores + out_rows[:, None] * TS_ROW_STRIDE + kcols[None, :],
                mask=row_mask[:, None] & kmask[None, :],
                other=-float("inf"),
            ).to(DTYPE)
            indices = tl.load(
                top_indices + out_rows[:, None] * TI_ROW_STRIDE + kcols[None, :],
                mask=row_mask[:, None] & kmask[None, :],
                other=0,
            ).to(tl.int32)
            lse_vals = tl.load(lse + out_rows, mask=row_mask, other=0.0).to(DTYPE)
            probs = tl.exp(scores - lse_vals[:, None])
            d_s = tl.load(
                d_scores + out_rows[:, None] * DS_ROW_STRIDE + kcols[None, :],
                mask=row_mask[:, None] & kmask[None, :],
                other=0.0,
            ).to(DTYPE)

            for vchunk in range(0, tl.cdiv(FEAT_V, FEATURE_BLOCK)):
                vcols = vchunk * FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)
                vcols_valid = vcols < FEAT_V
                do_tile = tl.load(
                    grad_out
                    + bh * GO_BH_STRIDE
                    + rows[:, None] * GO_ROW_STRIDE
                    + vcols[None, :],
                    mask=row_mask[:, None] & vcols_valid[None, :],
                    other=0.0,
                ).to(DTYPE)
                value_ptrs = (
                    d_values
                    + bh * VALUE_BH_STRIDE
                    + indices[:, :, None] * VALUE_ROW_STRIDE
                    + vcols[None, None, :]
                )
                tl.atomic_add(
                    value_ptrs,
                    probs[:, :, None] * do_tile[:, None, :],
                    mask=row_mask[:, None, None]
                    & kmask[:, None]
                    & vcols_valid[None, None, :],
                )

            for qchunk in range(0, tl.cdiv(FEAT_Q, FEATURE_BLOCK)):
                qcols = qchunk * FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)
                qcols_valid = qcols < FEAT_Q
                q_tile = tl.load(
                    queries
                    + bh * QUERY_BH_STRIDE
                    + rows[:, None] * QUERY_ROW_STRIDE
                    + qcols[None, :],
                    mask=row_mask[:, None] & qcols_valid[None, :],
                    other=0.0,
                ).to(DTYPE)
                key_ptrs = (
                    d_keys
                    + bh * KEY_BH_STRIDE
                    + indices[:, :, None] * KEY_ROW_STRIDE
                    + qcols[None, None, :]
                )
                tl.atomic_add(
                    key_ptrs,
                    d_s[:, :, None] * q_tile[:, None, :],
                    mask=row_mask[:, None, None]
                    & kmask[:, None]
                    & qcols_valid[None, None, :],
                )


class FusedTopKAttentionFunction(torch.autograd.Function):
    """Unified fused top-k dictionary attention with analytical backward.

    Forward fuses tiled affinity and the in-register running top-k before
    the existing gather-densify online-softmax aggregation, keeping the
    dense ``N x M`` matrix out of HBM entirely. Backward recomputes the
    softmax on the saved sparse support only. Inputs must be rank-4
    ``(B, H, N, d)`` / ``(B, H, M, d_k)`` / ``(B, H, M, d_v)`` float32 or
    float64 CUDA tensors; float16 stays on the reference path.
    """

    @staticmethod
    def forward(ctx, queries, keys, values, k):
        batch, heads, query_count, feat_q = queries.shape
        key_count = keys.shape[2]
        feat_v = values.shape[3]
        batch_heads = batch * heads
        device = queries.device
        dtype = queries.dtype

        queries2 = queries.reshape(batch_heads, query_count, feat_q)
        keys2 = keys.reshape(batch_heads, key_count, feat_q)
        values2 = values.reshape(batch_heads, key_count, feat_v)

        top_scores = torch.empty(
            (batch_heads, query_count, k), device=device, dtype=dtype
        )
        top_indices = torch.empty(
            (batch_heads, query_count, k), device=device, dtype=torch.int32
        )
        arr_len = max(FUSED_KEY_BLOCK, triton.next_power_of_2(k + FUSED_KEY_BLOCK))
        sort_stages = int(math.log2(arr_len))
        query_block = min(FUSED_QUERY_BLOCK, triton.next_power_of_2(query_count))
        feature_block = min(FUSED_FEATURE_BLOCK, triton.next_power_of_2(feat_q))
        # tl.dot has no float64 lowering; micro gradcheck shapes ride the
        # elementwise path regardless of size.
        use_dot = (
            min(query_block, arr_len, feature_block) >= 16
            and dtype != torch.float64
        )
        grid = (triton.cdiv(query_count, query_block), batch_heads)
        _fused_affinity_topk[grid](
            queries2,
            keys2,
            top_scores,
            top_indices,
            query_count,
            key_count,
            feat_q,
            queries2.stride(0),
            queries2.stride(1),
            keys2.stride(0),
            keys2.stride(1),
            top_scores.stride(1),
            top_scores.stride(2),
            k,
            arr_len,
            sort_stages,
            query_block,
            feature_block,
            FUSED_KEY_BLOCK,
            use_dot,
            key_count,
            _FUSED_DTYPE_MAP[dtype],
        )

        maxima = top_scores.max(dim=-1, keepdim=True).values
        lse = (maxima + torch.log(torch.exp(top_scores - maxima).sum(dim=-1, keepdim=True))).squeeze(-1)

        if dtype == torch.float64:
            broadcast_values = values2.unsqueeze(1).expand(
                batch_heads, query_count, key_count, feat_v
            )
            gathered = torch.gather(
                broadcast_values,
                2,
                top_indices.long().unsqueeze(-1).expand(
                    batch_heads, query_count, k, feat_v
                ),
            )
            weights = torch.softmax(top_scores, dim=-1)
            saved_out = (weights.unsqueeze(-1) * gathered).sum(dim=2)
        else:
            saved_out = torch.empty(
                (batch_heads, query_count, feat_v),
                device=device,
                dtype=torch.float32,
            )
            for index in range(batch_heads):
                saved_out[index] = tiled_topk_aggregate(
                    top_scores[index], top_indices[index], values2[index]
                )

        result = saved_out.view(batch, heads, query_count, feat_v)
        ctx.save_for_backward(
            queries2, keys2, values2, top_scores, top_indices, lse, saved_out
        )
        ctx.fused_geometry = (
            batch, heads, query_count, feat_q, key_count, feat_v, k
        )
        return result

    @staticmethod
    def backward(ctx, grad_output):
        queries2, keys2, values2, top_scores, top_indices, lse, saved_out = (
            ctx.saved_tensors
        )
        batch, heads, query_count, feat_q, key_count, feat_v, k = ctx.fused_geometry
        batch_heads = batch * heads
        dtype = saved_out.dtype
        grad_output2 = grad_output.reshape(batch_heads, query_count, feat_v)

        d_scores = torch.empty(
            (batch_heads, query_count, k), device=saved_out.device, dtype=dtype
        )
        dq = torch.empty(
            (batch_heads, query_count, feat_q), device=saved_out.device, dtype=dtype
        )
        d_keys = torch.zeros(
            (batch_heads, key_count, feat_q), device=saved_out.device, dtype=dtype
        )
        d_values = torch.zeros(
            (batch_heads, key_count, feat_v), device=saved_out.device, dtype=dtype
        )
        query_block = min(
            FUSED_BACKWARD_QUERY_BLOCK, triton.next_power_of_2(query_count)
        )
        feature_block = min(FUSED_FEATURE_BLOCK, triton.next_power_of_2(feat_q))
        grid = (triton.cdiv(query_count, query_block), batch_heads)
        kp = triton.next_power_of_2(k)
        kchunk = min(FUSED_KCHUNK, kp)
        _backward_dq[grid](
            queries2,
            keys2,
            values2,
            grad_output2,
            saved_out,
            top_scores,
            top_indices,
            lse,
            d_scores,
            dq,
            query_count,
            key_count,
            feat_q,
            feat_v,
            queries2.stride(0),
            queries2.stride(1),
            keys2.stride(0),
            keys2.stride(1),
            values2.stride(0),
            values2.stride(1),
            grad_output2.stride(0),
            grad_output2.stride(1),
            saved_out.stride(0),
            saved_out.stride(1),
            top_scores.stride(1),
            top_scores.stride(2),
            top_indices.stride(1),
            top_indices.stride(2),
            d_scores.stride(1),
            dq.stride(1),
            k,
            kp,
            kchunk,
            query_block,
            feature_block,
            _FUSED_DTYPE_MAP[dtype],
        )
        _backward_dkv[grid](
            queries2,
            grad_output2,
            top_scores,
            top_indices,
            lse,
            d_scores,
            d_keys,
            d_values,
            query_count,
            key_count,
            feat_q,
            feat_v,
            keys2.stride(0),
            keys2.stride(1),
            values2.stride(0),
            values2.stride(1),
            grad_output2.stride(0),
            grad_output2.stride(1),
            top_scores.stride(1),
            top_scores.stride(2),
            d_scores.stride(1),
            k,
            kp,
            kchunk,
            query_block,
            feature_block,
            _FUSED_DTYPE_MAP[dtype],
        )
        return (
            dq.view(batch, heads, query_count, feat_q),
            d_keys.view(batch, heads, key_count, feat_q),
            d_values.view(batch, heads, key_count, feat_v),
            None,
        )


def fused_topk_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Fused CUDA sparse top-k cross attention returning ``(B, H, N, d_v)``.

    Validates the same geometry as the reference path (matching leading
    ``(B, H)`` dims, contiguous feature dims, CUDA tensors), restricted to
    float32/float64 until fp16 backward is implemented. Raises ``ValueError``
    with actionable messages otherwise.
    """
    for name, tensor in (
        ("queries", queries),
        ("keys", keys),
        ("values", values),
    ):
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must be rank-4 (B, H, tokens/atoms, features); "
                f"got {tuple(tensor.shape)}"
            )
        if not tensor.is_floating_point():
            raise ValueError(
                f"{name} must have a floating-point dtype; got {tensor.dtype}"
            )
        if tensor.dtype not in (torch.float32, torch.float64):
            raise ValueError(_FUSED_DTYPE_MESSAGE)
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if queries.shape[0] != keys.shape[0] or queries.shape[1] != keys.shape[1]:
        raise ValueError("queries and keys must share leading (B, H) dimensions")
    if queries.shape[0:2] != values.shape[0:2]:
        raise ValueError(
            f"queries shape {tuple(queries.shape)} and values shape "
            f"{tuple(values.shape)} must share leading (B, H) dimensions"
        )
    if keys.shape[0:2] != values.shape[0:2]:
        raise ValueError(
            f"keys shape {tuple(keys.shape)} and values shape "
            f"{tuple(values.shape)} must share leading (B, H) dimensions"
        )
    if queries.shape[3] != keys.shape[3]:
        raise ValueError(
            f"query/key feature sizes must match; got "
            f"{queries.shape[3]} and {keys.shape[3]}"
        )
    if keys.shape[2] != values.shape[2]:
        raise ValueError(
            f"key/value atom counts must match; got "
            f"{keys.shape[2]} and {values.shape[2]}"
        )
    if queries.shape[0] == 0 or queries.shape[1] == 0:
        raise ValueError("batch and head dimensions must be non-empty")
    if queries.shape[2] == 0 or keys.shape[2] == 0 or queries.shape[3] == 0:
        raise ValueError(
            "query, key, and feature dimensions must be non-empty"
        )
    if values.shape[3] == 0:
        raise ValueError("value feature dimension must be non-empty")
    if (
        not isinstance(k, int)
        or isinstance(k, bool)
        or not 1 <= k <= keys.shape[2]
    ):
        raise ValueError(
            f"k must be an integer in [1, {keys.shape[2]}]; got {k}"
        )
    if queries.device.type != "cuda" or keys.device.type != "cuda" or values.device.type != "cuda":
        raise ValueError("fused path requires CUDA tensors")
    if queries.device != keys.device or queries.device != values.device:
        raise ValueError("queries, keys, and values must be on the same device")
    if triton is None:
        raise RuntimeError(
            "Triton is required for fused_topk_attention; install a "
            "Triton-enabled PyTorch environment"
        )
    return FusedTopKAttentionFunction.apply(queries, keys, values, k)
