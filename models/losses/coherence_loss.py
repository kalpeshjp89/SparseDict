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

    The hinge's ``1 / (M * (M - 1))`` normalization makes per-pair gradients
    negligible when only a few atom pairs are collinear, and the frame
    potential has an interior minimum that does not bound the max off-diagonal
    (mutual coherence). ``lambda_max`` adds a differentiable smooth-max limb

    ``L_max = lambda_max * tau * logsumexp(off_diagonal / tau)``

    whose gradient concentrates on the most collinear pairs via a softmax
    (temperature ``tau``; smaller tau sharpens toward the true max while
    spreading gradient over the near-max set). Default weight 0 keeps the
    historical behavior.
    """

    def __init__(
        self,
        lambda_fp: float = 1.0,
        lambda_hinge: float = 10.0,
        lambda_max: float = 0.0,
        max_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        for name, value in (
            ("lambda_fp", lambda_fp),
            ("lambda_hinge", lambda_hinge),
            ("lambda_max", lambda_max),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative; got {value}")
        if not math.isfinite(max_temperature) or max_temperature <= 0:
            raise ValueError(
                f"max_temperature must be finite and positive; got {max_temperature}"
            )
        self.lambda_fp = float(lambda_fp)
        self.lambda_hinge = float(lambda_hinge)
        self.lambda_max = float(lambda_max)
        self.max_temperature = float(max_temperature)

    def welch_bound(self, atom_count: int, atom_dim: int) -> float:
        """Return the Welch bound for ``atom_count`` atoms of dimension ``atom_dim``.

        For under-complete banks (``atom_count <= atom_dim``) an orthogonal
        frame exists, so the lower bound on the mutual coherence is 0.
        """
        if atom_count <= atom_dim:
            return 0.0
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

        loss = self.lambda_fp * frame_potential + self.lambda_hinge * hinge
        if self.lambda_max:
            smooth_max = self.max_temperature * torch.logsumexp(
                off_diagonal / self.max_temperature, dim=(-2, -1)
            )
            loss = loss + self.lambda_max * smooth_max
        return loss
