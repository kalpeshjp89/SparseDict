"""Compact MoE top-k dictionary model for image restoration smoke tests."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import hyperparams as hp
from models.losses.balance_loss import AuxiliaryLoadBalanceLoss
from models.modules.moe_dict import MoEDictionary
from models.modules.router import DegradationAwareRouter
from models.modules.topk_select import sparse_topk_attention


class MoETopKRestorer(nn.Module):
    """Restore RGB images using routed expert dictionaries and sparse attention.

    Inputs can be lower-resolution (e.g. bicubic 4x downsampled); provide
    ``output_size`` to interpolate them to the target size before processing.
    """

    def __init__(
        self,
        d: int = 64,
        m: int = 256,
        num_experts: int = 4,
        active_experts: int = 1,
        top_k: int = 32,
        reduction_ratio: int = 4,
        noise_std: float = 0.1,
        aux_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if m <= 1:
            raise ValueError(f"m must be greater than 1; got {m}")
        if not 1 <= top_k <= m:
            raise ValueError(f"top_k must be between 1 and m={m}; got {top_k}")
        if d <= 0 or reduction_ratio <= 0 or d % reduction_ratio:
            raise ValueError(
                f"d={d} must be positive and divisible by "
                f"reduction_ratio={reduction_ratio}"
            )

        self.d = int(d)
        self.m = int(m)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.d_k = self.d // reduction_ratio
        self.tau = float(hp.TAU_0.value) * math.log(self.m) / math.log(
            int(hp.M_0.value)
        )

        self.stem = nn.Conv2d(3, self.d, kernel_size=3, padding=1)
        self.query_projection = nn.Linear(self.d, self.d_k)
        self.router = DegradationAwareRouter(
            d=self.d,
            num_experts=self.num_experts,
            active_experts=active_experts,
            noise_std=noise_std,
        )
        self.dictionary = MoEDictionary(
            d=self.d,
            m=self.m,
            num_experts=self.num_experts,
            reduction_ratio=reduction_ratio,
        )
        self.attention_projection = nn.Linear(self.d, self.d)
        self.fusion = nn.Conv2d(self.d, self.d, kernel_size=3, padding=1)
        self.reconstruction = nn.Conv2d(self.d, 3, kernel_size=3, padding=1)
        self.balance_loss = AuxiliaryLoadBalanceLoss(weight=aux_weight)
        nn.init.normal_(self.reconstruction.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.reconstruction.bias)

    def forward(
        self,
        image: torch.Tensor,
        output_size: tuple[int, int] | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.size(1) != 3:
            raise ValueError(
                f"expected RGB image tensor (B, 3, H, W), got {tuple(image.shape)}"
            )
        if image.size(0) == 0 or image.size(-2) == 0 or image.size(-1) == 0:
            raise ValueError("batch and spatial dimensions must be non-empty")
        if output_size is not None:
            if len(output_size) != 2 or min(output_size) <= 0:
                raise ValueError(f"output_size must be positive (H, W); got {output_size}")
            image = F.interpolate(
                image, size=output_size, mode="bicubic", align_corners=False
            ).clamp(0, 1)

        features = self.stem(image)
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        gates = self.router(tokens, mode="token")
        queries = self.query_projection(tokens)
        keys, values = self.dictionary.project_experts()
        # Route weights bias atom selection toward the experts selected per token.
        scores = torch.einsum("bnk,eak->bnea", queries, keys)
        gate_logits = gates.clamp_min(torch.finfo(gates.dtype).tiny).log()
        scores = scores + gate_logits.unsqueeze(-1)
        scores = scores.masked_fill(gates.unsqueeze(-1) == 0, float("-inf"))
        scores = scores.flatten(start_dim=-2) / (
            self.tau * math.sqrt(self.d_k)
        )
        values = values.flatten(0, 1)
        attention = sparse_topk_attention(scores, values, self.top_k)
        attended = self.attention_projection(attention.output)
        attended = attended.transpose(1, 2).reshape(
            batch, channels, height, width
        )
        fused = features + F.gelu(self.fusion(attended))
        prediction = (image + self.reconstruction(F.gelu(fused))).clamp(0, 1)

        if return_aux:
            return prediction, self.balance_loss(gates)
        return prediction
