"""Expert-partitioned dictionary memory with per-expert key/value projections."""

from __future__ import annotations

import torch
import torch.nn as nn

from configs import hyperparams as hp


class MoEDictionary(nn.Module):
    """Store one dictionary and expose its contiguous expert-bank views.

    Dynamic residuals are supplied at call time so they can depend on the
    current input without duplicating the learned dictionary parameters.
    Projected keys and values are independent of query tokens and can be
    precomputed by callers when the dictionary and projections are frozen.
    """

    def __init__(
        self,
        d: int | None = None,
        m: int | None = None,
        num_experts: int = 4,
        reduction_ratio: int | None = None,
    ) -> None:
        super().__init__()
        self.d = int(hp.D.value if d is None else d)
        self.m = int(hp.M.value if m is None else m)
        self.num_experts = int(num_experts)
        self.reduction_ratio = int(
            hp.REDUCTION_RATIO.value
            if reduction_ratio is None
            else reduction_ratio
        )

        if self.d <= 0:
            raise ValueError(f"d must be positive; got {self.d}")
        if self.m <= 0:
            raise ValueError(f"m must be positive; got {self.m}")
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive; got {self.num_experts}")
        if self.m % self.num_experts != 0:
            raise ValueError(
                f"m={self.m} must be divisible by num_experts={self.num_experts}"
            )
        if self.reduction_ratio <= 0 or self.d % self.reduction_ratio != 0:
            raise ValueError(
                f"d={self.d} must be divisible by positive "
                f"reduction_ratio={self.reduction_ratio}"
            )

        self.d_k = self.d // self.reduction_ratio
        self.atoms_per_expert = self.m // self.num_experts
        self.dictionary = nn.Parameter(torch.empty(self.m, self.d))
        nn.init.xavier_uniform_(self.dictionary)

        self.w_k = nn.Parameter(torch.empty(self.num_experts, self.d, self.d_k))
        self.w_v = nn.Parameter(torch.empty(self.num_experts, self.d, self.d))
        for expert in range(self.num_experts):
            nn.init.xavier_uniform_(self.w_k[expert])
            nn.init.xavier_uniform_(self.w_v[expert])

    @property
    def dictionary_parameter_count(self) -> int:
        """Number of learned atom values, excluding projection parameters."""
        return self.dictionary.numel()

    def expert_dictionaries(
        self, residual: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return expert banks shaped (E, M/E, d), optionally refined."""
        banks = self.dictionary.view(
            self.num_experts, self.atoms_per_expert, self.d
        )
        if residual is None:
            return banks
        expected = (self.num_experts, self.atoms_per_expert, self.d)
        if tuple(residual.shape) != expected:
            raise ValueError(
                f"expected residual shape {expected}, got {tuple(residual.shape)}"
            )
        return banks + residual

    def project_experts(
        self, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project expert banks into keys (E, M/E, d_k) and values (E, M/E, d)."""
        banks = self.expert_dictionaries(residual)
        keys = torch.einsum("emd,edk->emk", banks, self.w_k)
        values = torch.einsum("emd,edv->emv", banks, self.w_v)
        return keys, values
