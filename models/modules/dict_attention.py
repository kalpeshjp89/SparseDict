"""Dense (un-sparsified) dictionary cross-attention — Phase 1 baseline.

Offline dictionary projections (independent of the query tokens X):
    K_D = D W_K    ∈ R^{M × d_k}
    V_D = D W_V    ∈ R^{M × d}

Query projection (depends on X):
    Q_X = X W_Q    ∈ R^{B × N × d_k}

Dense softmax attention:
    α = softmax(Q_X K_D^T / (τ √d_k))
    X_out = α V_D  ∈ R^{B × N × d}

τ = τ₀ · log(M_active) / log(M₀)  keeps selectivity stable as M grows.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import hyperparams as hp


class DenseDictCrossAttention(nn.Module):
    def __init__(
        self,
        d: int | None = None,
        reduction_ratio: int | None = None,
        m: int | None = None,
        tau_0: float | None = None,
        m_0: int | None = None,
        use_bias: bool | None = None,
    ) -> None:
        super().__init__()
        self.d = int(hp.D.value if d is None else d)
        self.reduction_ratio = int(
            hp.REDUCTION_RATIO.value if reduction_ratio is None else reduction_ratio
        )
        if self.d % self.reduction_ratio != 0:
            raise ValueError(
                f"d={self.d} must be divisible by r={self.reduction_ratio} so d_k is an integer"
            )
        self.d_k = self.d // self.reduction_ratio
        self.m = int(hp.M.value if m is None else m)
        self.tau_0 = float(hp.TAU_0.value if tau_0 is None else tau_0)
        self.m_0 = int(hp.M_0.value if m_0 is None else m_0)
        use_bias = bool(hp.USE_BIAS.value if use_bias is None else use_bias)

        # Dictionary atoms live in the same space as tokens (M × d).
        self.dictionary = nn.Parameter(torch.empty(self.m, self.d))
        nn.init.xavier_uniform_(self.dictionary)

        self.w_q = nn.Linear(self.d, self.d_k, bias=use_bias)
        self.w_k = nn.Linear(self.d, self.d_k, bias=use_bias)
        self.w_v = nn.Linear(self.d, self.d, bias=use_bias)

    def temperature(self, m_active: int | None = None) -> float:
        m_active = self.m if m_active is None else m_active
        if m_active <= 1:
            raise ValueError(f"M_active must be > 1 for log(M); got {m_active}")
        if self.m_0 <= 1:
            raise ValueError(f"M_0 must be > 1 for log(M_0); got {self.m_0}")
        return self.tau_0 * math.log(m_active) / math.log(self.m_0)

    def project_dictionary(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Offline K_D, V_D: depend on D, W_K, W_V only — not on X."""
        k_d = self.w_k(self.dictionary)  # M × d_k
        v_d = self.w_v(self.dictionary)  # M × d
        return k_d, v_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: query tokens, shape (B, N, d)
        Returns:
            x_out: (B, N, d)
        """
        if x.ndim != 3 or x.size(-1) != self.d:
            raise ValueError(f"expected X of shape (B, N, {self.d}), got {tuple(x.shape)}")

        q_x = self.w_q(x)  # B × N × d_k
        k_d, v_d = self.project_dictionary()

        tau = self.temperature(k_d.size(0))
        scale = tau * math.sqrt(self.d_k)
        # (B, N, d_k) @ (d_k, M) -> (B, N, M)
        logits = torch.matmul(q_x, k_d.transpose(0, 1)) / scale
        alpha = F.softmax(logits, dim=-1)
        # (B, N, M) @ (M, d) -> (B, N, d)
        return torch.matmul(alpha, v_d)
