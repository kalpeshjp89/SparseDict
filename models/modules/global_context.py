"""Global context stem for non-local degradation conditioning.

Downsamples a full-resolution image to a small thumbnail and extracts a
global vector v_global in R^d that captures image-level statistics
(atmospheric light, global illumination, haze density).  This vector is
concatenated to the router input and used for FiLM modulation so that
local 64x64 patches receive global context without running the full
transformer backbone on high-resolution images.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalContextStem(nn.Module):
    """Lightweight 3-layer ConvNet: 64x64 thumbnail -> global vector R^d."""

    def __init__(self, d: int, thumbnail_size: int = 64) -> None:
        super().__init__()
        self.d = int(d)
        self.thumbnail_size = int(thumbnail_size)
        self.conv1 = nn.Conv2d(3, d // 2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d // 2, d, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(d, d, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(d, d)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return v_global of shape (B, d).

        ``image`` is (B, 3, H, W) at any resolution; it is downsampled
        internally to (thumbnail_size, thumbnail_size).
        """
        thumb = F.interpolate(
            image, size=(self.thumbnail_size, self.thumbnail_size),
            mode="bilinear", align_corners=False,
        )
        x = F.gelu(self.conv1(thumb))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = self.pool(x).flatten(1)
        return self.proj(x)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: gamma * X + beta, conditioned on v_global."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(d, d)
        self.beta_proj = nn.Linear(d, d)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, v_global: torch.Tensor) -> torch.Tensor:
        """Modulate features x (B, C, H, W) with v_global (B, d)."""
        gamma = self.gamma_proj(v_global).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_proj(v_global).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta
