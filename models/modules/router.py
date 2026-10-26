"""Image-level and token-level expert routing for dictionary memories."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import hyperparams as hp


class DegradationAwareRouter(nn.Module):
    """Route token features to a sparse, normalized subset of experts."""

    def __init__(
        self,
        d: int | None = None,
        num_experts: int = 4,
        active_experts: int = 1,
        hidden_dim: int | None = None,
        noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.d = int(hp.D.value if d is None else d)
        self.num_experts = int(num_experts)
        self.active_experts = int(active_experts)
        self.hidden_dim = self.d if hidden_dim is None else int(hidden_dim)
        self.noise_std = float(noise_std)

        if self.d <= 0:
            raise ValueError(f"d must be positive; got {self.d}")
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive; got {self.num_experts}")
        if not 1 <= self.active_experts <= self.num_experts:
            raise ValueError(
                "active_experts must be between 1 and num_experts; "
                f"got {self.active_experts} and {self.num_experts}"
            )
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive; got {self.hidden_dim}")
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be non-negative; got {self.noise_std}")

        self.fc1 = nn.Linear(self.d, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_experts)

    def forward(self, x: torch.Tensor, mode: str = "token") -> torch.Tensor:
        """Return sparse gates shaped (B, E) or (B, N, E).

        ``x`` is a token tensor shaped (B, N, d). In ``image`` mode tokens
        are mean-pooled before routing; in ``token`` mode each token is routed
        independently. Gaussian logit jitter is applied only during training.
        """
        if x.ndim != 3 or x.size(-1) != self.d:
            raise ValueError(f"expected X of shape (B, N, {self.d}), got {tuple(x.shape)}")
        if x.size(0) == 0 or x.size(1) == 0:
            raise ValueError("batch and token dimensions must be non-empty")
        if mode == "image":
            features = x.mean(dim=1)
        elif mode == "token":
            features = x
        else:
            raise ValueError(f"mode must be 'image' or 'token'; got {mode!r}")

        logits = self.fc2(F.gelu(self.fc1(features)))
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        selected_logits, selected_experts = logits.topk(
            self.active_experts, dim=-1
        )
        selected_gates = F.softmax(selected_logits, dim=-1)
        return torch.zeros_like(logits).scatter(
            dim=-1, index=selected_experts, src=selected_gates
        )
