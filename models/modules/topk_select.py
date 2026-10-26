"""Hard top-k affinity selection and sparse dictionary-value aggregation."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F


class TopKAttentionResult(NamedTuple):
    output: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor
    masked_scores: torch.Tensor


class _StraightThroughTopK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        scores: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_scores, indices = torch.topk(scores, k, dim=-1)
        ctx.save_for_backward(indices)
        ctx.input_shape = scores.shape
        ctx.mark_non_differentiable(indices)
        return selected_scores, indices

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_selected_scores: torch.Tensor | None,
        grad_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, None]:
        del grad_indices
        if grad_selected_scores is None:
            return None, None
        (indices,) = ctx.saved_tensors
        grad_scores = grad_selected_scores.new_zeros(ctx.input_shape).scatter(
            dim=-1, index=indices, src=grad_selected_scores
        )
        return grad_scores, None


def straight_through_topk(
    scores: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sorted top-k scores/indices with selected-only STE gradients."""
    _validate_scores(scores, k)
    return _StraightThroughTopK.apply(scores, k)


def gumbel_perturbed_topk(
    scores: torch.Tensor,
    k: int,
    sigma: float = 1.0,
    num_samples: int = 2,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Monte Carlo top-k score/index samples under Gumbel perturbation.

    The leading output dimension is the sample dimension. Callers can average
    losses over it to form the Monte Carlo estimate. Gradients flow to base
    scores only for entries selected in each corresponding sample.
    """
    _validate_scores(scores, k)
    if sigma < 0 or not math.isfinite(sigma):
        raise ValueError(f"sigma must be finite and non-negative; got {sigma}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive; got {num_samples}")

    epsilon = torch.finfo(scores.dtype).eps
    sample_scores = []
    sample_indices = []
    for _ in range(num_samples):
        uniform = torch.rand(
            scores.shape,
            dtype=scores.dtype,
            device=scores.device,
            generator=generator,
        ).clamp(min=epsilon, max=1.0 - epsilon)
        gumbel_noise = -torch.log(-torch.log(uniform))
        selected_scores, indices = straight_through_topk(
            scores + sigma * gumbel_noise, k
        )
        sample_scores.append(selected_scores)
        sample_indices.append(indices)
    return torch.stack(sample_scores), torch.stack(sample_indices)


def hard_topk_mask(
    scores: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Set all but the top-k scores to negative infinity along the last axis."""
    _validate_scores(scores, k)
    selected_scores, indices = torch.topk(scores, k, dim=-1)
    masked_scores = torch.full_like(scores, float("-inf")).scatter(
        dim=-1, index=indices, src=selected_scores
    )
    return masked_scores, indices


def sparse_topk_attention(
    scores: torch.Tensor,
    values: torch.Tensor,
    k: int,
    selection: str = "hard",
) -> TopKAttentionResult:
    """Softmax top-k scores and aggregate only the selected value rows.

    ``scores`` has shape ``(..., Q, M)``. ``values`` may be shared with shape
    ``(M, D)`` or batch-specific with shape ``(..., M, D)`` matching the
    leading dimensions of ``scores``.
    """
    if selection == "hard":
        masked_scores, indices = hard_topk_mask(scores, k)
    elif selection == "ste":
        selected_scores, indices = straight_through_topk(scores, k)
        masked_scores = torch.full_like(scores, float("-inf")).scatter(
            dim=-1, index=indices, src=selected_scores
        )
    else:
        raise ValueError(f"selection must be 'hard' or 'ste'; got {selection!r}")
    if values.ndim < 2:
        raise ValueError(
            f"expected values with atom and feature dimensions, got {tuple(values.shape)}"
        )
    if values.size(-2) != scores.size(-1):
        raise ValueError(
            f"value atom count {values.size(-2)} must match score width {scores.size(-1)}"
        )
    if values.device != scores.device:
        raise ValueError("scores and values must be on the same device")

    batch_shape = scores.shape[:-2]
    value_batch_shape = values.shape[:-2]
    if value_batch_shape and tuple(value_batch_shape) != tuple(batch_shape):
        raise ValueError(
            "values must be shared across batches or have leading dimensions "
            f"{tuple(batch_shape)}; got {tuple(value_batch_shape)}"
        )

    selected_weights = F.softmax(
        masked_scores.gather(dim=-1, index=indices), dim=-1
    )
    weights = torch.zeros_like(scores).scatter(
        dim=-1, index=indices, src=selected_weights
    )

    query_count = scores.size(-2)
    expanded_values = values.unsqueeze(-3).expand(
        *batch_shape, query_count, values.size(-2), values.size(-1)
    )
    gather_indices = indices.unsqueeze(-1).expand(
        *indices.shape, values.size(-1)
    )
    selected_values = expanded_values.gather(dim=-2, index=gather_indices)
    output = (selected_values * selected_weights.unsqueeze(-1)).sum(dim=-2)
    return TopKAttentionResult(output, weights, indices, masked_scores)


def _validate_scores(scores: torch.Tensor, k: int) -> None:
    if scores.ndim < 2:
        raise ValueError(
            f"expected scores with query and atom dimensions, got {tuple(scores.shape)}"
        )
    if not scores.is_floating_point():
        raise ValueError("scores must have a floating-point dtype")
    if not 1 <= k <= scores.size(-1):
        raise ValueError(f"k must be between 1 and {scores.size(-1)}; got {k}")
