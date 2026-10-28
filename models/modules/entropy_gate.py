"""Adaptive entropy-gated dynamic top-k routing (Task 6.3)."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_ENTROPY_EPS = 1e-12


class EntropyGateResult(NamedTuple):
    output: torch.Tensor
    gate: torch.Tensor
    k: torch.Tensor
    entropy: torch.Tensor
    threshold: torch.Tensor | None


class EntropyGate(nn.Module):
    """Gate tokens between the dynamic top-k path and a linear bypass.

    Each token's ``C``-d features project to a ``V``-entry vocabulary and
    normalize through softmax into a distribution whose Shannon entropy
    ``H_i`` measures degeneracy or unpredictability. Tokens above the
    running threshold
    ``tau_H = mu_H + kappa * sigma_H``
    stay on the dynamic top-k attention path, while tokens below it are
    replaced by a learned linear ``W_skip`` bypass. The selected budget is
    adaptive: ``k_i`` scales linearly in ``H_i`` from ``k_min`` to
    ``k_max``, using the maximal uniform entropy ``log(V)`` as reference.
    """

    def __init__(
        self,
        channels: int = 64,
        vocab_size: int = 256,
        kappa: float = 0.2,
        k_min: int = 8,
        k_max: int = 64,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive; got {channels}")
        if vocab_size < 2:
            raise ValueError(
                f"vocab_size must be at least two for a non-degenerate "
                f"entropy; got {vocab_size}"
            )
        if not math.isfinite(kappa) or kappa <= 0:
            raise ValueError(f"kappa must be finite and positive; got {kappa}")
        if not 1 <= k_min <= k_max:
            raise ValueError(
                "k_min and k_max must satisfy 1 <= k_min <= k_max; got "
                f"{k_min} and {k_max}"
            )
        if not 0 <= momentum <= 1:
            raise ValueError(f"momentum must be in [0, 1]; got {momentum}")

        self.channels = int(channels)
        self.vocab_size = int(vocab_size)
        self.kappa = float(kappa)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.momentum = float(momentum)
        self.max_entropy = math.log(int(vocab_size))

        self.vocab_proj = nn.Linear(self.channels, self.vocab_size, bias=False)
        self.skip_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.register_buffer("entropy_mean", torch.zeros(()))
        self.register_buffer("entropy_std", torch.zeros(()))
        self.register_buffer(
            "entropy_stats_initialized", torch.zeros((), dtype=torch.bool)
        )

    def shannon_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Return the nats entropy of ``softmax(logits)`` along the last axis."""
        probs = F.softmax(logits, dim=-1)
        safe = probs.clamp_min(_ENTROPY_EPS)
        return -(probs * safe.log()).sum(dim=-1)

    def _update_statistics(self, entropy: torch.Tensor) -> None:
        flat = entropy.detach().flatten()
        if flat.numel() == 0:
            return
        batch_mean = flat.mean()
        batch_std = flat.std(unbiased=False) if flat.numel() > 1 else flat.new_zeros(())
        if not bool(self.entropy_stats_initialized):
            self.entropy_mean.copy_(batch_mean)
            self.entropy_std.copy_(batch_std)
            self.entropy_stats_initialized.fill_(True)
            return
        momentum = self.momentum
        self.entropy_mean.mul_(momentum).add_(batch_mean, alpha=1.0 - momentum)
        self.entropy_std.mul_(momentum).add_(batch_std, alpha=1.0 - momentum)

    def forward(self, features: torch.Tensor) -> EntropyGateResult:
        if features.ndim != 3:
            raise ValueError(
                f"expected (B, N, {self.channels}) features; "
                f"got {tuple(features.shape)}"
            )
        if features.size(0) == 0 or features.size(1) == 0:
            raise ValueError("batch and token dimensions must be non-empty")
        if features.size(-1) != self.channels:
            raise ValueError(
                f"expected {self.channels} feature channels; got {features.size(-1)}"
            )
        if bool(torch.isinf(features).any()) or bool(torch.isnan(features).any()):
            raise ValueError("features must contain only finite values")

        entropy = self.shannon_entropy(self.vocab_proj(features))
        if self.training:
            self._update_statistics(entropy)

        threshold = None
        if bool(self.entropy_stats_initialized):
            threshold = self.entropy_mean + self.kappa * self.entropy_std
            gate = (entropy > threshold).to(features.dtype)
        else:
            flat = entropy.detach().flatten()
            batch_mean = flat.mean()
            batch_std = (
                flat.std(unbiased=False) if flat.numel() > 1 else flat.new_zeros(())
            )
            threshold = batch_mean + self.kappa * batch_std
            # Degenerate guard: identical entropies give a mean that can round
            # up or down by one ulp (making sigma vanishingly small but
            # nonzero), so the strict comparison is unreliable. A zero-spread
            # batch carries no routing signal -> all bypassed.
            spread = flat.max() - flat.min()
            if float(spread) > 0.0:
                gate = (entropy > threshold).to(features.dtype)
            else:
                gate = torch.zeros_like(entropy, dtype=features.dtype)

        normalized = entropy.clamp(max=self.max_entropy) / self.max_entropy
        k = self.k_min + (self.k_max - self.k_min) * normalized
        k = k.round().clamp(min=self.k_min, max=self.k_max).to(torch.long)

        output = gate.unsqueeze(-1) * features + (
            1.0 - gate
        ).unsqueeze(-1) * self.skip_proj(features)
        return EntropyGateResult(output, gate, k, entropy, threshold)
