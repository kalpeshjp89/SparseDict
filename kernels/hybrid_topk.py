"""Hybrid top-k attention: cuBLAS affinity + Triton top-k gather.

Option 3 hybrid path for Task 5.4. The dense score matrix chunks
``S_chunk = Q @ K_chunk^T`` are produced by cuBLAS (never the full
``(N, M)`` tensor at once), each score chunk is reduced to its exact
chunk-local top-``k`` with CUDA ``torch.topk``, and a final Triton kernel
bitonically selects the global top-k, normalizes scores, gathers values,
and writes the attention output.

Exactness of ``c == k`` per chunk: a chunk's global-top-``k`` elements
are a subset of that chunk's local top-``k`` list, so the union of
chunk-local top-``k`` lists contains the global top-``k``, and one final
bitonic sort of the union recovers it exactly. Chunks shorter than
``k`` (possible only for non-multiple-of-``HYBRID_CHUNK`` key counts)
store ``-inf`` padded entries; they lose the final merge unless the row
genuinely has fewer than ``k`` finite scores, and their index slots
carry a sentinel that the merge maps to ``PAD_INDEX = key_count`` (out
of gather range) while the value stays ``-inf``, so downstream softmax
weights are exactly zero. The autograd wrapper consumes the returned
top-k support for its analytical sparse backward.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# Score-column width reduced to local top-k candidates.
HYBRID_CHUNK = 1024
# Score columns computed by one cuBLAS launch (multiple HYBRID_CHUNKs).
HYBRID_GEMM_CHUNK = 1024

_HYBRID_DTYPE_MESSAGE = "hybrid path supports float32 inputs"


if triton is not None:
    @triton.jit
    def fused_topk_gather_softmax_kernel(
        cand_scores,
        cand_indices,
        top_scores,
        top_indices,
        lse,
        value_dict,
        output,
        QUERY_COUNT: tl.constexpr,
        KEY_COUNT: tl.constexpr,
        VALUE_DIM: tl.constexpr,
        K: tl.constexpr,
        CAND_ROW_STRIDE: tl.constexpr,
        CAND_COUNT: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        CHUNK_PAD_INDEX: tl.constexpr,
        PAD_INDEX: tl.constexpr,
        TS_ROW_STRIDE: tl.constexpr,
        TI_ROW_STRIDE: tl.constexpr,
        VALUE_BH_STRIDE: tl.constexpr,
        VALUE_ROW_STRIDE: tl.constexpr,
        OUTPUT_ROW_STRIDE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        SORT_STAGES: tl.constexpr,
        K_BLOCK: tl.constexpr,
        VALUE_BLOCK: tl.constexpr,
    ):
        """Select global top-k, normalize, gather, and aggregate one query."""
        row = tl.program_id(0)
        positions = tl.arange(0, BLOCK_SIZE)
        valid = positions < CAND_COUNT
        scores = tl.load(
            cand_scores + row * CAND_ROW_STRIDE + positions,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)
        local_idx = tl.load(
            cand_indices + row * CAND_ROW_STRIDE + positions,
            mask=valid,
            other=CHUNK_PAD_INDEX,
        )
        chunk_base = (positions // K) * CHUNK_SIZE
        indices = tl.where(
            valid & (local_idx != CHUNK_PAD_INDEX),
            chunk_base + local_idx,
            PAD_INDEX,
        ).to(tl.int32)

        for stage in range(1, SORT_STAGES + 1):
            stage_mask = 1 << stage
            for stride_log in range(stage - 1, -1, -1):
                stride = 1 << stride_log
                partner_positions = positions ^ stride
                partner_scores = tl.gather(scores, partner_positions, axis=0)
                partner_indices = tl.gather(indices, partner_positions, axis=0)
                self_is_better = (scores > partner_scores) | (
                    (scores == partner_scores) & (indices < partner_indices)
                )
                lower_in_pair = (positions & stride) == 0
                ascending_group = (positions & stage_mask) != 0
                take_better = lower_in_pair != ascending_group
                choose_self = self_is_better == take_better
                scores = tl.where(choose_self, scores, partner_scores)
                indices = tl.where(choose_self, indices, partner_indices)

        top_positions = tl.arange(0, K_BLOCK)
        top_scores_row = tl.gather(scores, top_positions, axis=0)
        top_indices_row = tl.gather(indices, top_positions, axis=0)
        selected = top_positions < K
        selected_scores = tl.where(selected, top_scores_row, -float("inf"))
        row_max = tl.max(selected_scores, axis=0)
        weights = tl.exp(selected_scores - row_max)
        weight_sum = tl.sum(weights, axis=0)
        row_lse = row_max + tl.log(weight_sum)

        features = tl.arange(0, VALUE_BLOCK)
        selected_indices = tl.where(selected, top_indices_row, 0)
        gathered = tl.load(
            value_dict
            + (row // QUERY_COUNT) * VALUE_BH_STRIDE
            + selected_indices[:, None] * VALUE_ROW_STRIDE
            + features[None, :],
            mask=selected[:, None]
            & (top_indices_row[:, None] < KEY_COUNT)
            & (features[None, :] < VALUE_DIM),
            other=0.0,
        )
        result = tl.sum((weights / weight_sum)[:, None] * gathered, axis=0)

        tl.store(
            top_scores + row * TS_ROW_STRIDE + top_positions,
            top_scores_row,
            mask=selected,
        )
        tl.store(
            top_indices + row * TI_ROW_STRIDE + top_positions,
            top_indices_row,
            mask=selected,
        )
        tl.store(lse + row, row_lse)
        tl.store(
            output + row * OUTPUT_ROW_STRIDE + features,
            result,
            mask=features < VALUE_DIM,
        )


def _hybrid_topk_attention_with_aux(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    k: int,
    score_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hybrid output, selected support, and log-sum-exp for autograd."""
    if not all(torch.is_tensor(tensor) for tensor in (queries, keys, values)):
        raise TypeError("queries, keys, and values must be torch tensors")
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        raise ValueError("queries/keys/values must be rank-4 (B, H, N, D)")
    if not all(tensor.is_contiguous() for tensor in (queries, keys, values)):
        raise ValueError("queries, keys, and values must be contiguous")
    if queries.device.type != "cuda" or keys.device != queries.device or values.device != queries.device:
        raise ValueError("queries, keys, and values must share a CUDA device")
    if any(tensor.dtype != torch.float32 for tensor in (queries, keys, values)):
        raise ValueError(_HYBRID_DTYPE_MESSAGE)
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if triton is None:
        raise RuntimeError(
            "hybrid path requires a Triton-enabled PyTorch environment"
        )

    batch, heads, query_count, feat_q = queries.shape
    key_count = keys.shape[2]
    feat_v = values.shape[3]
    if k >= key_count:
        raise ValueError(f"hybrid selector requires k in [1, {key_count - 1}]")
    if (batch, heads) != keys.shape[:2] or (batch, heads) != values.shape[:2]:
        raise ValueError("queries, keys, and values must share batch and head dimensions")
    if queries.shape[3] != keys.shape[3]:
        raise ValueError("query and key feature dimensions must match")
    if key_count != values.shape[2]:
        raise ValueError("key and value atom counts must match")
    if min(batch, heads, query_count, key_count, feat_q, feat_v) <= 0:
        raise ValueError("batch, token, key, and feature dimensions must be non-empty")

    batch_heads = batch * heads
    candidate_rows = batch_heads * query_count
    queries2 = queries.reshape(batch_heads, query_count, feat_q)
    keys2 = keys.reshape(batch_heads, key_count, feat_q)
    values2 = values.reshape(batch_heads, key_count, feat_v)

    chunk = max(HYBRID_CHUNK, triton.next_power_of_2(k))
    gemm_chunk = max(HYBRID_GEMM_CHUNK, chunk)
    gemm_chunk = triton.cdiv(gemm_chunk, chunk) * chunk
    total_chunks = triton.cdiv(key_count, chunk)
    candidate_count = total_chunks * k
    candidate_scores = torch.empty(
        (candidate_rows, candidate_count), device=queries.device, dtype=torch.float32
    )
    candidate_indices = torch.empty(
        (candidate_rows, candidate_count), device=queries.device, dtype=torch.int32
    )
    candidate_row_stride = candidate_scores.stride(0)
    for base in range(0, key_count, gemm_chunk):
        stop = min(base + gemm_chunk, key_count)
        width = stop - base
        chunks_per_row = triton.cdiv(width, chunk)
        score_chunk = torch.matmul(
            queries2, keys2[:, base:stop].transpose(-2, -1)
        ).reshape(candidate_rows, width)
        score_chunk.mul_(score_scale)
        for chunk_index in range(chunks_per_row):
            local_start = chunk_index * chunk
            local_stop = min(local_start + chunk, width)
            local_k = min(k, local_stop - local_start)
            local_scores, local_indices = torch.topk(
                score_chunk[:, local_start:local_stop],
                local_k,
                dim=-1,
                largest=True,
                sorted=False,
            )
            slot_start = (base // chunk + chunk_index) * k
            candidate_scores[:, slot_start : slot_start + local_k].copy_(
                local_scores
            )
            candidate_indices[:, slot_start : slot_start + local_k].copy_(
                local_indices.to(torch.int32)
            )
            if local_k < k:
                candidate_scores[:, slot_start + local_k : slot_start + k].fill_(
                    -float("inf")
                )
                candidate_indices[:, slot_start + local_k : slot_start + k].fill_(
                    chunk
                )

    top_scores = torch.empty(
        (candidate_rows, k), device=queries.device, dtype=torch.float32
    )
    top_indices = torch.empty(
        (candidate_rows, k), device=queries.device, dtype=torch.int32
    )
    lse = torch.empty((candidate_rows,), device=queries.device, dtype=torch.float32)
    output = torch.empty(
        (candidate_rows, feat_v), device=queries.device, dtype=torch.float32
    )
    merge_block = triton.next_power_of_2(candidate_count)
    k_block = triton.next_power_of_2(k)
    value_block = triton.next_power_of_2(feat_v)
    fused_topk_gather_softmax_kernel[(candidate_rows,)](
        candidate_scores,
        candidate_indices,
        top_scores,
        top_indices,
        lse,
        values2,
        output,
        query_count,
        key_count,
        feat_v,
        k,
        candidate_row_stride,
        candidate_count,
        chunk,
        chunk,
        key_count,
        top_scores.stride(0),
        top_indices.stride(0),
        values2.stride(0),
        values2.stride(1),
        output.stride(0),
        merge_block,
        int(math.log2(merge_block)),
        k_block,
        value_block,
        num_warps=4,
    )
    return (
        top_scores.reshape(batch_heads, query_count, k),
        top_indices.reshape(batch_heads, query_count, k),
        lse.reshape(batch_heads, query_count),
        output.reshape(batch_heads, query_count, feat_v),
    )


def hybrid_topk_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    k: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Hybrid forward using chunked cuBLAS scores and a fused Triton gather."""
    if not torch.is_tensor(queries):
        raise TypeError("queries, keys, and values must be torch tensors")
    if not torch.is_tensor(keys) or keys.ndim != 4:
        raise ValueError("keys must be rank-4 (B, H, M, D)")
    if queries.ndim != 4:
        raise ValueError("queries must be rank-4 (B, H, N, D)")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError("temperature must be a positive finite number")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number")
    if k == keys.shape[2]:
        from kernels.fused_topk_attn import fused_topk_attention

        return fused_topk_attention(queries, keys, values, k, temperature)
    score_scale = 1.0 / (temperature * math.sqrt(queries.shape[-1]))
    output = _hybrid_topk_attention_with_aux(
        queries, keys, values, k, score_scale
    )[-1]
    return output.reshape(*queries.shape[:3], values.shape[-1])
