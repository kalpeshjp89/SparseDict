"""Auxiliary coefficient-of-variation loss for expert routing balance."""

from __future__ import annotations

import torch
import torch.nn as nn

from configs import hyperparams as hp


class AuxiliaryLoadBalanceLoss(nn.Module):
    """Penalize imbalance in router importance and selected expert load.

    ``gates`` must contain normalized, sparse routing weights with experts in
    the final dimension; any preceding dimensions are treated as routed items.
    Population standard deviation is used for a stable, batch-size-independent
    coefficient of variation.
    """

    def __init__(self, weight: float = 0.01) -> None:
        super().__init__()
        self.weight = float(weight)
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative; got {self.weight}")

    def forward(self, gates: torch.Tensor) -> torch.Tensor:
        if gates.ndim < 2:
            raise ValueError(
                f"expected gates with item and expert dimensions, got {tuple(gates.shape)}"
            )
        if gates.size(-1) < 2:
            raise ValueError("at least two experts are required")

        flat_gates = gates.reshape(-1, gates.size(-1))
        if flat_gates.size(0) == 0:
            raise ValueError("at least one routed item is required")
        if not torch.isfinite(flat_gates).all():
            raise ValueError("gates must contain only finite values")
        if torch.any(flat_gates < 0):
            raise ValueError("gates must be non-negative")
        if torch.any(flat_gates.sum(dim=-1) <= 0):
            raise ValueError("each routed item must have positive total gate weight")

        importance = flat_gates.sum(dim=0)
        load = flat_gates.ne(0).sum(dim=0).to(dtype=flat_gates.dtype)
        importance_cv_sq = (
            importance.std(correction=0) / importance.mean()
        ).square()
        load_cv_sq = (load.std(correction=0) / load.mean()).square()
        return self.weight * (importance_cv_sq + load_cv_sq)
