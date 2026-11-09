"""Spatial-frequency dual-domain MoE dictionary (Task 6.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.router import DegradationAwareRouter
from models.modules.topk_select import sparse_topk_attention

_SPECTRAL_EPS = 1e-8


def haar_dwt2(x: torch.Tensor):
    """Decompose ``(B, C, H, W)`` into Haar subbands ``{LL, LH, HL, HH}``.

    Each subband has shape ``(B, C, H/2, W/2)``. ``H`` and ``W`` must be
    even. The 2x2 block of ``x`` at rows ``(2i, 2i+1)`` and columns
    ``(2j, 2j+1)`` contributes ``(a + b + c + d) / 2`` to ``LL`` and the
    corresponding signed combinations to ``LH``, ``HL``, and ``HH``, so the
    four subbands carry an orthonormal Haar transform scaled by two.
    """
    if x.ndim != 4:
        raise ValueError(f"expected a rank-4 image; got {tuple(x.shape)}")
    batch, channels, height, width = x.shape
    if height % 2 or width % 2:
        raise ValueError(
            f"haar_dwt2 requires even height and width; got ({height}, {width})"
        )
    if height == 0 or width == 0 or channels == 0 or batch == 0:
        raise ValueError("haar_dwt2 requires non-empty dimensions")
    blocks = x.reshape(batch, channels, height // 2, 2, width // 2, 2)
    tl_ = blocks[:, :, :, 0, :, 0]
    tr_ = blocks[:, :, :, 0, :, 1]
    bl_ = blocks[:, :, :, 1, :, 0]
    br_ = blocks[:, :, :, 1, :, 1]
    ll = (tl_ + tr_ + bl_ + br_) / 2.0
    lh = (tl_ + tr_ - bl_ - br_) / 2.0
    hl = (tl_ - tr_ + bl_ - br_) / 2.0
    hh = (tl_ - tr_ - bl_ + br_) / 2.0
    return ll, lh, hl, hh


def haar_idwt2(ll, lh, hl, hh):
    """Reconstruct ``(B, C, H, W)`` from the four Haar subbands."""
    for band, name in ((ll, "LL"), (lh, "LH"), (hl, "HL"), (hh, "HH")):
        if band.ndim != 4:
            raise ValueError(f"{name} subband must be rank-4; got {tuple(band.shape)}")
    batch, channels, hb, wb = ll.shape
    blocks = torch.empty(
        batch, channels, hb, 2, wb, 2, device=ll.device, dtype=ll.dtype
    )
    blocks[..., 0, :, 0] = (ll + lh + hl + hh) / 2.0
    blocks[..., 0, :, 1] = (ll + lh - hl - hh) / 2.0
    blocks[..., 1, :, 0] = (ll - lh + hl - hh) / 2.0
    blocks[..., 1, :, 1] = (ll - lh - hl + hh) / 2.0
    return blocks.reshape(batch, channels, hb * 2, wb * 2)


class DualDomainMoEDictionary(nn.Module):
    """Partition ``E=4`` expert dictionaries over spatial and spectral paths.

    Expert banks: 0 and 1 hold spatial edge and texture atoms, expert 2 holds
    wavelet low-pass (``LL``) structural atoms, and expert 3 holds
    high-frequency atoms applied to a Fourier-domain pathway built from the
    amplitude spectrum. A scale-conditioned four-way router selects active
    banks; branch outputs are reconstructed with the inverse transforms and
    combined by channel-gated ``1x1`` convolution fusion.
    """

    def __init__(
        self,
        channels: int = 64,
        atoms_per_expert: int = 64,
        reduction_ratio: int = 4,
        top_k: int = 16,
        noise_std: float = 0.1,
        global_context_dim: int = 0,
    ) -> None:
        super().__init__()
        if channels <= 0 or channels % reduction_ratio:
            raise ValueError(
                f"channels={channels} must be positive and divisible by "
                f"reduction_ratio={reduction_ratio}"
            )
        if atoms_per_expert <= 0:
            raise ValueError(
                f"atoms_per_expert must be positive; got {atoms_per_expert}"
            )
        self.channels = int(channels)
        self.atoms_per_expert = int(atoms_per_expert)
        self.d_k = self.channels // int(reduction_ratio)
        self.top_k = int(top_k)
        self.global_context_dim = int(global_context_dim)

        # Atom rows are C-dimensional; banks [0, 1] are spatial, [2] is
        # wavelet low-pass, and [3] is Fourier high-frequency.
        self.dictionary = nn.Parameter(
            torch.randn(4, self.atoms_per_expert, self.channels) * 0.02
        )
        self.spatial_router = DegradationAwareRouter(
            d=self.channels,
            num_experts=4,
            active_experts=1,
            noise_std=noise_std,
            global_context_dim=global_context_dim,
        )

        domains = ("spatial", "wavelet", "fourier")
        self.projections = nn.ModuleList(
            nn.ModuleDict(
                {
                    "query": nn.Linear(self.channels, self.d_k, bias=False),
                    "key": nn.Linear(self.channels, self.d_k, bias=False),
                    "value": nn.Linear(self.channels, self.channels, bias=False),
                }
            )
            for _ in domains
        )
        self.fusion_conv = nn.Conv2d(3 * self.channels, self.channels, kernel_size=1)
        self.gate_conv = nn.Conv2d(3 * self.channels, self.channels, kernel_size=1)

    @property
    def dictionary_parameter_count(self) -> int:
        return self.dictionary.numel()

    def _atom_projections(self, branch: int, bank: torch.Tensor):
        """Return ``(keys, values)`` projections of the atom rows.

        ``bank`` is ``(E, M, C)`` with atoms as rows. ``keys`` is
        ``(E, M, d_k)`` and ``values`` is ``(E, M, C)``.
        """
        proj = self.projections[branch]
        keys = torch.einsum("emc,kc->emk", bank, proj["key"].weight)
        values = torch.einsum("emc,fc->emf", bank, proj["value"].weight)
        return keys, values

    def _mixture(
        self,
        query_tokens: torch.Tensor,
        branch: int,
        bank: torch.Tensor,
        top_k: int,
        gates: torch.Tensor | None = None,
    ):
        """Softmax top-k mixture over the bank's projected atoms.

        ``query_tokens`` is ``(B, N, C)``; the returned tokens are
        ``(B, N, C)``. Optional ``gates`` is ``(B, N, E)`` at the same device
        and dtype as the atom projections; a zero gate disables that expert
        for the token, and nonzero gates are folded in additively, so slots
        selected by the top-k come out renormalized together.
        """
        proj = self.projections[branch]
        q = proj["query"](query_tokens)  # (B, N, d_k)
        keys, values = self._atom_projections(branch, bank)
        scores = torch.einsum("bnk,emk->bnem", q, keys)
        if gates is not None:
            batch, tokens, experts = scores.shape[:3]
            if tuple(gates.shape) != (batch, tokens, experts):
                raise ValueError(
                    f"expected gates of shape {(batch, tokens, experts)}; "
                    f"got {tuple(gates.shape)}"
                )
            tiny = torch.finfo(scores.dtype).tiny
            log_gates = gates.clamp_min(tiny).log().unsqueeze(-1)
            scores = scores + log_gates
            scores = scores.masked_fill(gates.unsqueeze(-1) == 0, float("-inf"))
        batch, tokens, experts, atoms = scores.shape
        scores = scores.reshape(batch, tokens, experts * atoms)
        flat_values = values.reshape(experts * atoms, values.size(-1))
        result = sparse_topk_attention(scores, flat_values, top_k)
        return result.output

    def forward(
        self,
        feature_map: torch.Tensor,
        return_gates: bool = False,
        scale_factor: int = 1,
        global_context: torch.Tensor | None = None,
    ):
        if feature_map.ndim != 4:
            raise ValueError(
                f"expected a rank-4 feature map; got {tuple(feature_map.shape)}"
            )
        batch, channels, height, width = feature_map.shape
        if channels != self.channels:
            raise ValueError(
                f"expected {self.channels} channels; got {channels}"
            )
        if batch == 0:
            raise ValueError("batch dimension must be non-empty")
        if height < 2 or width < 2 or height % 2 or width % 2:
            raise ValueError(
                "spatial dimensions must be even and at least two; got "
                f"({height}, {width})"
            )
        if scale_factor not in (1, 2, 3, 4):
            raise ValueError(
                f"scale_factor must be one of 1, 2, 3, or 4; got {scale_factor}"
            )

        if scale_factor == 2:
            active_atoms = max(1, self.atoms_per_expert // 2)
        elif scale_factor == 3:
            active_atoms = max(1, (3 * self.atoms_per_expert) // 4)
        else:
            active_atoms = self.atoms_per_expert

        # --- Spatial pathway: routed mixture of the two edge/texture banks. ---
        tokens = feature_map.flatten(2).transpose(1, 2)
        spatial_bank = self.dictionary[:2, :active_atoms]
        gates = self.spatial_router(
            tokens, mode="token", scale_factor=scale_factor,
            global_context=global_context,
        )
        spatial_gates = gates[..., :2]
        has_spatial_route = spatial_gates.sum(dim=-1, keepdim=True) > 0
        fallback_gates = torch.zeros_like(spatial_gates)
        fallback_gates[..., 0] = 1
        safe_spatial_gates = torch.where(
            has_spatial_route, spatial_gates, fallback_gates
        )
        sp_tokens = self._mixture(
            tokens,
            branch=0,
            bank=spatial_bank,
            top_k=min(self.top_k, 2 * active_atoms),
            gates=safe_spatial_gates,
        )
        sp_map = sp_tokens.transpose(1, 2).reshape(batch, channels, height, width)
        sp_map = sp_map * has_spatial_route.transpose(1, 2).reshape(
            batch, 1, height, width
        )

        # --- Wavelet pathway: attention over the LL subband structure. ---
        ll, lh, hl, hh = haar_dwt2(feature_map)
        ll_tokens = ll.flatten(2).transpose(1, 2)
        low_bank = self.dictionary[2:3, :active_atoms]
        ll_tokens_out = self._mixture(
            ll_tokens,
            branch=1,
            bank=low_bank,
            top_k=min(self.top_k, active_atoms),
        )
        ll_map = ll_tokens_out.transpose(1, 2).reshape(
            batch, channels, height // 2, width // 2
        )
        wave_map = haar_idwt2(ll_map, lh, hl, hh)
        gate_map = gates.transpose(1, 2).reshape(batch, 4, height, width)
        wave_gate = F.adaptive_avg_pool2d(
            gate_map[:, 2:3], (height // 2, width // 2)
        )
        wave_map = wave_map * F.interpolate(
            wave_gate, size=(height, width), mode="nearest"
        )

        # --- Fourier pathway: attention over amplitude spectrum tokens. ---
        spectrum = torch.fft.rfft2(feature_map, norm="backward")
        amplitude = torch.sqrt(
            spectrum.real.pow(2) + spectrum.imag.pow(2) + _SPECTRAL_EPS
        )
        phase = torch.atan2(spectrum.imag, spectrum.real)  # fixed carrier
        spec_tokens = amplitude.flatten(2).transpose(1, 2)
        high_bank = self.dictionary[3:4, :active_atoms]
        amp_tokens_out = self._mixture(
            spec_tokens,
            branch=2,
            bank=high_bank,
            top_k=min(self.top_k, active_atoms),
        )
        amp_map = amp_tokens_out.transpose(1, 2).reshape(
            batch, channels, height, width // 2 + 1
        )
        fourier_gate = F.interpolate(
            gate_map[:, 3:4],
            size=(height, width // 2 + 1),
            mode="bilinear",
            align_corners=False,
        )
        amp_map = amp_map * fourier_gate
        fourier_spectrum = torch.polar(amp_map, phase)
        fourier_map = torch.fft.irfft2(
            fourier_spectrum, s=(height, width), norm="backward"
        )

        fusion = torch.cat((sp_map, wave_map, fourier_map), dim=1)
        gate = torch.sigmoid(self.gate_conv(fusion))
        output = self.fusion_conv(fusion) * gate
        if return_gates:
            return output, gates
        return output
