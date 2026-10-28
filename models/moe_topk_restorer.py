"""Compact MoE top-k dictionary model for image restoration smoke tests."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import hyperparams as hp
from models.losses.balance_loss import AuxiliaryLoadBalanceLoss
from models.losses.coherence_loss import MutualCoherenceLoss
from models.modules.dual_domain_moe import DualDomainMoEDictionary
from models.modules.entropy_gate import EntropyGate
from models.modules.moe_dict import MoEDictionary
from models.modules.router import DegradationAwareRouter
from models.modules.topk_select import sparse_topk_attention


class ParallelDilatedAffineHead(nn.Module):
    """Predict per-token scale and shift from parallel multi-scale context."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim < 2:
            raise ValueError(f"hidden_dim must be at least 2; got {hidden_dim}")
        self.hidden_dim = int(hidden_dim)
        self.branches = nn.ModuleList(
            nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=3,
                padding=rate,
                dilation=rate,
                groups=self.hidden_dim,
            )
            for rate in (1, 2, 4)
        )
        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.hidden_dim * len(self.branches), self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim * 2),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
        with torch.no_grad():
            self.proj[-1].bias[: self.hidden_dim].fill_(1.0)

    def forward(
        self, tokens: torch.Tensor, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.size(-1) != self.hidden_dim:
            raise ValueError(
                f"expected tokens with shape (B, N, {self.hidden_dim}), "
                f"got {tuple(tokens.shape)}"
            )
        if height <= 0 or width <= 0 or tokens.size(1) != height * width:
            raise ValueError(
                f"token count {tokens.size(1)} must equal height*width "
                f"({height}*{width})"
            )

        batch = tokens.size(0)
        feature_map = tokens.transpose(1, 2).reshape(
            batch, self.hidden_dim, height, width
        )
        multi_scale = torch.cat(
            [branch(feature_map) for branch in self.branches], dim=1
        )
        contextual_tokens = multi_scale.flatten(2).transpose(1, 2)
        params = self.proj(contextual_tokens)
        return params.chunk(2, dim=-1)


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
        self.affine_head = ParallelDilatedAffineHead(hidden_dim=self.d)
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
        gamma, beta = self.affine_head(tokens, height, width)
        modulated = gamma * attention.output + beta
        attended = self.attention_projection(modulated)
        attended = attended.transpose(1, 2).reshape(
            batch, channels, height, width
        )
        fused = features + F.gelu(self.fusion(attended))
        prediction = (image + self.reconstruction(F.gelu(fused))).clamp(0, 1)

        if return_aux:
            return prediction, self.balance_loss(gates)
        return prediction


# --- Task 6.4: Master Restorer Architecture Integration -------------------

def _charbonnier(prediction, target, eps=1e-6):
    """Element-wise Charbonnier penalty ``sqrt(d^2 + eps)``."""
    delta = prediction - target
    return (delta * delta + eps).sqrt()


def charbonnier_loss(prediction, target):
    """Mean Charbonnier reconstruction loss (robust L1 surrogate)."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} does not match "
            f"target shape {tuple(target.shape)}"
        )
    return _charbonnier(prediction, target).mean()


def gradient_difference_loss(prediction, target, eps=1e-6):
    """Charbonnier penalty on horizontal and vertical image gradients."""
    if prediction.shape[-2] < 2 or prediction.shape[-1] < 2:
        raise ValueError("gradients require spatial dims of at least 2")

    def _dx(x):
        return x[..., :, 1:] - x[..., :, :-1]

    def _dy(x):
        return x[..., 1:, :] - x[..., :-1, :]

    horizontal = _charbonnier(_dx(prediction), _dx(target), eps).mean()
    vertical = _charbonnier(_dy(prediction), _dy(target), eps).mean()
    return horizontal + vertical


def mutual_coherence_of(atoms: torch.Tensor) -> torch.Tensor:
    """Maximum off-diagonal pairwise cosine among atom rows of ``(..., M, d)``."""
    if atoms.ndim < 2:
        raise ValueError(f"expected atom rows (..., M, d); got {tuple(atoms.shape)}")
    flat = atoms.flatten(end_dim=-2)
    if flat.size(0) < 2:
        raise ValueError("at least two atoms are required")
    normalized = flat / (flat.norm(dim=-1, keepdim=True) + 1e-12)
    gram = normalized @ normalized.transpose(-1, -2)
    mask = 1.0 - torch.eye(flat.size(0), device=flat.device, dtype=gram.dtype)
    return (gram * mask).abs().max()


class MasterMoETopKRestorer(nn.Module):
    """Master restorer unifying Phase-6 modules with a structured loss dict.

    Pipeline: stem -> entropy gate (bypass mixed tokens + pass-through
    mask) -> dual-domain MoE dictionaries (routed spatial banks + wavelet
    LL + Fourier amplitude) -> parallel dilated affine modulation ->
    residual fusion and reconstruction head.

    Returns ``(I_rec, loss_dict)`` where ``loss_dict`` holds differentiable
    scalars: ``charbonnier``, ``gradient_difference``, ``load_balance``,
    and ``coherence``. Pass ``total_loss=True`` to get a pre-summed total
    as the second element instead.
    """

    def __init__(
        self,
        d: int = 64,
        atoms_per_expert: int = 64,
        reduction_ratio: int = 4,
        top_k: int = 16,
        noise_std: float = 0.1,
        aux_weight: float = 0.01,
        coherence_lambda_fp: float = 1.0,
        coherence_lambda_hinge: float = 10.0,
        use_fused_kernel: bool = False,
    ) -> None:
        super().__init__()
        if d <= 0 or reduction_ratio <= 0 or d % reduction_ratio:
            raise ValueError(
                f"d={d} must be positive and divisible by "
                f"reduction_ratio={reduction_ratio}"
            )
        if not 1 <= top_k <= 2 * atoms_per_expert:
            raise ValueError(
                f"top_k must be between 1 and 2*atoms_per_expert="
                f"{2 * atoms_per_expert}; got {top_k}"
            )
        self.d = int(d)
        self.top_k = int(top_k)
        self.d_k = self.d // reduction_ratio
        self.use_fused_kernel = bool(use_fused_kernel)

        self.stem = nn.Conv2d(3, self.d, kernel_size=3, padding=1)
        self.entropy_gate = EntropyGate(
            channels=self.d,
            vocab_size=256,
            kappa=0.2,
            k_min=min(8, self.top_k),
            k_max=min(64, 2 * atoms_per_expert),
        )
        self.dual_domain = DualDomainMoEDictionary(
            channels=self.d,
            atoms_per_expert=atoms_per_expert,
            reduction_ratio=reduction_ratio,
            top_k=self.top_k,
            noise_std=noise_std,
        )
        self.affine_head = ParallelDilatedAffineHead(hidden_dim=self.d)
        self.attention_projection = nn.Linear(self.d, self.d)
        self.fusion = nn.Conv2d(self.d, self.d, kernel_size=3, padding=1)
        self.reconstruction = nn.Conv2d(self.d, 3, kernel_size=3, padding=1)
        self.balance_loss = AuxiliaryLoadBalanceLoss(weight=aux_weight)
        self.coherence_loss = MutualCoherenceLoss(
            lambda_fp=coherence_lambda_fp,
            lambda_hinge=coherence_lambda_hinge,
        )
        nn.init.normal_(self.reconstruction.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.reconstruction.bias)

    def _effective_top_k(self, adaptive_k: torch.Tensor) -> int:
        """Collapse per-token k_i to one selectable integer budget."""
        k_min = int(self.entropy_gate.k_min)
        k_max = int(self.entropy_gate.k_max)
        mean_k = float(adaptive_k.to(torch.float32).mean().round())
        return int(min(max(mean_k, k_min), max(k_min, self.top_k)))

    def forward(
        self,
        image: torch.Tensor,
        reference: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
        total_loss: bool = False,
    ):
        if image.ndim != 4 or image.size(1) != 3:
            raise ValueError(
                f"expected RGB image tensor (B, 3, H, W), got {tuple(image.shape)}"
            )
        if image.size(0) == 0 or image.size(-2) == 0 or image.size(-1) == 0:
            raise ValueError("batch and spatial dimensions must be non-empty")
        if self.use_fused_kernel:
            raise RuntimeError(
                "use_fused_kernel=True requires the Task 5.4b "
                "FusedTopKAttentionFunction Triton kernel on CUDA, which is "
                "not yet implemented; run with the reference path instead"
            )
        if output_size is not None:
            if len(output_size) != 2 or min(output_size) <= 0:
                raise ValueError(
                    f"output_size must be positive (H, W); got {output_size}"
                )
            image = F.interpolate(
                image, size=output_size, mode="bicubic", align_corners=False
            ).clamp(0, 1)

        target = image if reference is None else reference
        if tuple(target.shape) != tuple(image.shape):
            raise ValueError(
                f"reference shape {tuple(target.shape)} must match the "
                f"processed image shape {tuple(image.shape)}"
            )

        features = self.stem(image)
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)

        gate_result = self.entropy_gate(tokens)
        gated_tokens = gate_result.output

        attention_input = gated_tokens.transpose(1, 2).reshape(
            batch, channels, height, width
        )
        attention_map, gates = self.dual_domain(
            attention_input, return_gates=True
        )
        attention_tokens = attention_map.flatten(2).transpose(1, 2)

        gamma, beta = self.affine_head(tokens, height, width)
        modulated = gamma * attention_tokens + beta
        attended = self.attention_projection(modulated)
        attended = attended.transpose(1, 2).reshape(
            batch, channels, height, width
        )

        fused = features + F.gelu(self.fusion(attended))
        reconstruction_input = F.gelu(fused)
        prediction = (image + self.reconstruction(reconstruction_input)).clamp(0, 1)

        losses = {
            "charbonnier": charbonnier_loss(prediction, target),
            "gradient_difference": gradient_difference_loss(prediction, target),
            "load_balance": self.balance_loss(gates),
            "coherence": self.coherence_loss(
                self.dual_domain.dictionary.flatten(end_dim=-2)
            ),
        }
        return prediction, sum(losses.values()) if total_loss else losses
