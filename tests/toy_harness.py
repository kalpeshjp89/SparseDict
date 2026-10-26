"""Generate small synthetic inputs for fail-fast model checks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def generate_toy_tensors(
    batch_size: int = 2,
    channels: int = 3,
    height: int = 64,
    width: int = 64,
    atoms: int = 256,
    dim: int = 64,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an image batch (B, C, H, W) and dictionary (M, d)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    low_quality = torch.rand(
        batch_size, channels, height, width, generator=generator
    )
    dictionary = torch.randn(atoms, dim, generator=generator)
    return low_quality, dictionary


def main() -> int:
    low_quality, dictionary = generate_toy_tensors()
    expected_image_shape = (2, 3, 64, 64)
    expected_dictionary_shape = (256, 64)
    print(f"X shape: {tuple(low_quality.shape)} (expected {expected_image_shape})")
    print(f"D shape: {tuple(dictionary.shape)} (expected {expected_dictionary_shape})")
    return int(
        tuple(low_quality.shape) != expected_image_shape
        or tuple(dictionary.shape) != expected_dictionary_shape
    )


if __name__ == "__main__":
    raise SystemExit(main())
