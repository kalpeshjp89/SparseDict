"""Low-coherence dictionary regularization (Task 6.1)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

_NORMALIZATION_EPS = 1e-8


class MutualCoherenceLoss(nn.Module):
    """Frame potential plus a Welch-bound hinge over the atom Gram matrix.

    Atoms are the rows of ``dictionary`` (shape ``(..., M, d)``). Rows are
    unit-normalized and combined into a Gram matrix ``G`` whose off-diagonal
    entries measure atom collinearity. The loss penalizes deviations from
    orthogonality and, via a hinge, any pairwise absolute cosine exceeding
    the Welch bound
    ``mu_WB = sqrt((M - d) / (d * (M - 1)))``.
    """

    def __init__(self, lambda_fp: float = 1.0, lambda_hinge: float = 10.0) -> None:
        super().__init__()
        for name, value in (("lambda_fp", lambda_fp), ("lambda_hinge", lambda_hinge)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative; got {value}")
        self.lambda_fp = float(lambda_fp)
        self.lambda_hinge = float(lambda_hinge)

    def welch_bound(self, atom_count: int, atom_dim: int) -> float:
        """Return the Welch bound for ``atom_count`` atoms of dimension ``atom_dim``."""
        return math.sqrt((atom_count - atom_dim) / (atom_dim * (atom_count - 1)))

    def forward(self, dictionary: torch.Tensor) -> torch.Tensor:
        if not dictionary.is_floating_point():
            raise ValueError("dictionary must have a floating-point dtype")
        atom_count = dictionary.size(-2)
        atom_dim = dictionary.size(-1)
        if atom_count < 2:
            raise ValueError(
                f"the Welch bound requires at least two atoms; got {atom_count}"
            )
        if atom_count < atom_dim:
            raise ValueError(
                "the Welch bound requires atom_count >= atom_dim; got "
                f"M={atom_count}, d={atom_dim}"
            )
        if bool(torch.isinf(dictionary).any()) or bool(torch.isnan(dictionary).any()):
            raise ValueError("dictionary must contain only finite values")

        atoms = dictionary / (dictionary.norm(dim=-1, keepdim=True) + _NORMALIZATION_EPS)
        identity = torch.eye(
            atom_count, device=dictionary.device, dtype=dictionary.dtype
        )
        gram = atoms @ atoms.transpose(-1, -2)
        frame_potential = ((gram - identity) ** 2).sum(dim=(-2, -1))

        mask = 1.0 - identity
        off_diagonal = (gram * mask).abs()
        welch = self.welch_bound(atom_count, atom_dim)
        excess = torch.clamp(off_diagonal - welch, min=0.0)
        hinge = excess.pow(2).sum(dim=(-2, -1)) / (atom_count * (atom_count - 1))

        return self.lambda_fp * frame_potential + self.lambda_hinge * hinge
